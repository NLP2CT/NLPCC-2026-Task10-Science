"""Task message construction, response parsing, and fallback predictions."""

from __future__ import annotations

import base64
import importlib.util
import json
import mimetypes
import re
from pathlib import Path
from typing import Any

from baseline_prompting.errors import BaselineError
from baseline_prompting.labels import TRACK1_LABELS, TRACK2_LABELS


PROMPT_DIR = Path(__file__).with_name("prompts")


def load_prompt_template(name: str) -> str:
    return (PROMPT_DIR / name).read_text(encoding="utf-8").strip()


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        stripped = fence.group(1)
    else:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end != -1 and end > start:
            stripped = stripped[start : end + 1]
    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise BaselineError(f"model did not return a JSON object: {text[:500]}") from exc
    if not isinstance(obj, dict):
        raise BaselineError("model JSON response must be an object")
    return obj


def validate_label(label: Any, allowed: list[str], field: str) -> str:
    if not isinstance(label, str) or label not in allowed:
        raise BaselineError(f"invalid {field}: {label!r}; allowed={allowed}")
    return label


def track1_sentences(record: dict[str, Any]) -> list[str]:
    sentence_label = record.get("sentence_label")
    if not isinstance(sentence_label, list):
        raise BaselineError("Track 1 input must contain sentence_label list with sentence text")
    sentences: list[str] = []
    for index, item in enumerate(sentence_label, 1):
        if not isinstance(item, dict) or not isinstance(item.get("sentence"), str):
            raise BaselineError(f"Track 1 sentence_label[{index}] must contain sentence string")
        sentences.append(item["sentence"])
    return sentences


def image_data_url(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.name)
    if not mime_type:
        mime_type = "image/jpeg"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{payload}"


def resolve_image_path(img_path: Any, image_root: str | Path) -> Path | None:
    if not isinstance(img_path, str) or not img_path:
        return None
    path = Path(img_path)
    if path.is_absolute() and path.exists():
        return path
    root_path = Path(image_root) / path
    if root_path.exists():
        return root_path
    if path.exists():
        return path
    return None


def evidence_caption(evidence: dict[str, Any]) -> str:
    captions: list[str] = []
    for key in ("text", "table_caption", "image_caption"):
        value = evidence.get(key)
        if isinstance(value, str):
            captions.append(value)
        elif isinstance(value, list):
            captions.extend(str(item) for item in value if item is not None)
    return "\n".join(captions)


def build_track1_messages(
    record: dict[str, Any],
    image_mode: str,
    image_root: str | Path,
) -> list[dict[str, Any]]:
    sentences = track1_sentences(record)
    evidence_bundle = record.get("evidence_bundle")
    if not isinstance(evidence_bundle, list):
        raise BaselineError("Track 1 input must contain evidence_bundle list")

    evidence_items: list[dict[str, Any]] = []
    for index, evidence in enumerate(evidence_bundle, 1):
        if not isinstance(evidence, dict):
            raise BaselineError(f"Track 1 evidence_bundle[{index}] must be an object")
        evidence_items.append(
            {
                "index": index,
                "type": evidence.get("type", "unknown"),
                "caption_or_text": evidence_caption(evidence),
                "img_path": evidence.get("img_path", ""),
            }
        )

    user_prompt = load_prompt_template("track1_user.txt").format(
        num_sentences=len(sentences),
        claim_text=record.get("claim_text", ""),
        sentences_json=json.dumps(
            [{"index": i, "sentence": sentence} for i, sentence in enumerate(sentences, 1)],
            ensure_ascii=False,
            indent=2,
        ),
        evidence_json=json.dumps(evidence_items, ensure_ascii=False, indent=2),
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
    for evidence_item, evidence in zip(evidence_items, evidence_bundle):
        content.append({"type": "text", "text": f"Attached image for evidence item #{evidence_item['index']} follows if available."})
        if image_mode == "data-url":
            path = resolve_image_path(evidence.get("img_path"), image_root)
            if path is not None:
                content.append({"type": "image_url", "image_url": {"url": image_data_url(path)}})
    return [
        {"role": "system", "content": load_prompt_template("track1_system.txt")},
        {"role": "user", "content": content},
    ]


def parse_track1_response(text: str, num_sentences: int) -> list[str]:
    obj = extract_json_object(text)
    labels = obj.get("labels")
    if not isinstance(labels, list):
        raise BaselineError("Track 1 response must contain labels list")
    if len(labels) != num_sentences:
        raise BaselineError(f"Track 1 response label count mismatch: expected {num_sentences}, got {len(labels)}")
    return [validate_label(label, TRACK1_LABELS, "Track 1 label") for label in labels]


def iter_track2_paragraphs(record: dict[str, Any]) -> list[tuple[str, str]]:
    full_text = record.get("cited_paper_full_text")
    if not isinstance(full_text, list):
        raise BaselineError("Track 2 input must contain cited_paper_full_text list")
    paragraphs: list[tuple[str, str]] = []
    for index, item in enumerate(full_text, 1):
        if not isinstance(item, dict) or len(item) != 1:
            raise BaselineError(f"Track 2 cited_paper_full_text[{index}] must be a single-key object")
        para_id, text = next(iter(item.items()))
        if not isinstance(para_id, str) or not isinstance(text, str):
            raise BaselineError(f"Track 2 cited_paper_full_text[{index}] must map string id to string text")
        paragraphs.append((para_id, text))
    return paragraphs


def approximate_token_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z]+|\d+(?:\.\d+)?|[^\sA-Za-z\d]", text))


def count_tokens(text: str, model: str | None = None) -> int:
    if importlib.util.find_spec("tiktoken") is not None:
        try:
            import tiktoken  # type: ignore

            if model:
                try:
                    encoding = tiktoken.encoding_for_model(model)
                except KeyError:
                    encoding = tiktoken.get_encoding("cl100k_base")
            else:
                encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception:
            pass
    return approximate_token_count(text)


def truncate_track2_paragraphs_by_tokens(
    record: dict[str, Any],
    max_input_tokens: int,
    model: str | None = None,
) -> tuple[list[tuple[str, str]], bool]:
    paragraphs = iter_track2_paragraphs(record)
    claim_text = str(record.get("claim_text", ""))
    used = count_tokens(claim_text, model=model)
    selected: list[tuple[str, str]] = []
    truncated = False
    for para_id, text in paragraphs:
        rendered = f"\n{para_id}: {text}"
        para_tokens = count_tokens(rendered, model=model)
        if selected and used + para_tokens > max_input_tokens:
            truncated = True
            break
        if not selected and used + para_tokens > max_input_tokens:
            selected.append((para_id, text))
            truncated = True
            break
        selected.append((para_id, text))
        used += para_tokens
    return selected, truncated or len(selected) < len(paragraphs)


def build_track2_messages(
    record: dict[str, Any],
    max_input_tokens: int,
    model: str | None = None,
) -> tuple[list[dict[str, Any]], set[str]]:
    paragraphs, truncated = truncate_track2_paragraphs_by_tokens(record, max_input_tokens, model=model)
    para_ids = {para_id for para_id, _ in paragraphs}
    paper_text = "\n".join(f"{para_id}: {text}" for para_id, text in paragraphs)
    note = ""
    if truncated:
        note = "Only the included prefix of the cited paper is provided due to context limits.\n"
    prompt = load_prompt_template("track2_user.txt").format(
        labels=", ".join(TRACK2_LABELS),
        note=note,
        claim_text=record.get("claim_text", ""),
        paper_text=paper_text,
    )
    return [
        {"role": "system", "content": load_prompt_template("track2_system.txt")},
        {"role": "user", "content": prompt},
    ], para_ids


def parse_track2_response(text: str, allowed_para_ids: set[str]) -> tuple[str, list[str]]:
    obj = extract_json_object(text)
    label = validate_label(obj.get("label"), TRACK2_LABELS, "Track 2 label")
    evidence = obj.get("evidence_para_ids", [])
    if evidence is None:
        evidence = []
    if not isinstance(evidence, list):
        raise BaselineError("Track 2 response evidence_para_ids must be a list")
    result: list[str] = []
    seen: set[str] = set()
    for value in evidence:
        if not isinstance(value, str):
            raise BaselineError("Track 2 evidence_para_ids must contain strings")
        if value in seen:
            continue
        if value not in allowed_para_ids:
            raise BaselineError(f"Track 2 evidence id not present in prompt: {value}")
        seen.add(value)
        result.append(value)
        if len(result) == 3:
            break
    return label, result


def fallback_prediction(track: int, record: dict[str, Any], sample_id_value: str) -> dict[str, Any]:
    if track == 1:
        count = len(track1_sentences(record))
        return {"id": sample_id_value, "labels": ["Supported"] * count}
    return {"id": sample_id_value, "label": "Supported", "evidence_para_ids": []}
