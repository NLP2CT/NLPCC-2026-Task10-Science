"""Batch baseline runner."""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from baseline_prompting.client import OpenAICompatibleClient
from baseline_prompting.errors import BaselineError
from baseline_prompting.io_utils import append_jsonl, load_jsonl, read_completed_ids, sample_id
from baseline_prompting.tasks import (
    build_track1_messages,
    build_track2_messages,
    fallback_prediction,
    parse_track1_response,
    parse_track2_response,
    track1_sentences,
)


def write_error(path: str | Path | None, sample_id_value: str, exc: Exception) -> None:
    if not path:
        return
    error_path = Path(path)
    error_path.parent.mkdir(parents=True, exist_ok=True)
    append_jsonl({"id": sample_id_value, "error": str(exc)}, error_path)


def read_error_ids(path: str | Path | None) -> set[str]:
    if not path:
        return set()
    error_path = Path(path)
    if not error_path.exists():
        return set()
    ids: set[str] = set()
    for record in load_jsonl(error_path):
        value = record.get("id")
        if isinstance(value, str) and value:
            ids.add(value)
    return ids


def remove_ids_from_jsonl(path: str | Path, ids: set[str]) -> None:
    if not ids:
        return
    output_path = Path(path)
    if not output_path.exists():
        return
    kept = [record for record in load_jsonl(output_path) if record.get("id") not in ids]
    with output_path.open("w", encoding="utf-8") as f:
        for record in kept:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def reset_error_log(path: str | Path | None) -> None:
    if not path:
        return
    error_path = Path(path)
    error_path.parent.mkdir(parents=True, exist_ok=True)
    error_path.write_text("", encoding="utf-8")


def predict_record(
    record: dict[str, Any],
    track: int,
    sample_id_value: str,
    client: Any,
    model: str,
    temperature: float,
    max_output_tokens: int,
    track1_image_mode: str,
    image_root: str | Path,
    track2_max_input_tokens: int,
) -> dict[str, Any]:
    if track == 1:
        messages = build_track1_messages(record, track1_image_mode, image_root)
        content = client.chat_completion(messages, model, temperature, max_output_tokens)
        labels = parse_track1_response(content, len(track1_sentences(record)))
        return {"id": sample_id_value, "labels": labels}

    messages, allowed_para_ids = build_track2_messages(record, track2_max_input_tokens, model=model)
    content = client.chat_completion(messages, model, temperature, max_output_tokens)
    label, evidence_para_ids = parse_track2_response(content, allowed_para_ids)
    return {"id": sample_id_value, "label": label, "evidence_para_ids": evidence_para_ids}


def predict_with_retries(
    record: dict[str, Any],
    track: int,
    sample_id_value: str,
    client: Any,
    model: str,
    temperature: float,
    max_output_tokens: int,
    track1_image_mode: str,
    image_root: str | Path,
    track2_max_input_tokens: int,
    retries: int,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return predict_record(
                record=record,
                track=track,
                sample_id_value=sample_id_value,
                client=client,
                model=model,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                track1_image_mode=track1_image_mode,
                image_root=image_root,
                track2_max_input_tokens=track2_max_input_tokens,
            )
        except Exception as exc:
            last_error = exc
            if attempt >= retries:
                raise
            time.sleep(min(2**attempt, 8))
    raise last_error or BaselineError("prediction failed")


def _handle_prediction_error(args: argparse.Namespace, record: dict[str, Any], sid: str, exc: Exception) -> dict[str, Any]:
    write_error(args.error_log, sid, exc)
    print(f"[error] {sid}: {exc}", file=sys.stderr, flush=True)
    if args.on_error == "raise":
        raise exc
    return fallback_prediction(args.track, record, sid)


def run_baseline(args: argparse.Namespace, client: Any | None = None) -> int:
    records = load_jsonl(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = read_completed_ids(output_path) if args.resume else set()
    existing_prediction_count = len(completed)
    retry_ids = read_error_ids(args.error_log) if args.resume else set()
    if retry_ids:
        completed -= retry_ids
        remove_ids_from_jsonl(output_path, retry_ids)
    reset_error_log(args.error_log)
    if not args.resume:
        output_path.write_text("", encoding="utf-8")

    if client is None:
        client = OpenAICompatibleClient(args.base_url, args.api_key, timeout=args.timeout)

    total = len(records)
    if args.limit is not None:
        records = records[: args.limit]

    workers = max(1, int(getattr(args, "workers", 1) or 1))
    written = 0
    denominator = min(total, len(records))
    indexed_records = [
        (index, record, sample_id(record, args.track, index))
        for index, record in enumerate(records, 1)
        if sample_id(record, args.track, index) not in completed
    ]
    print(
        f"Resume: existing_predictions={existing_prediction_count} "
        f"retry_errors={len(retry_ids)} skipped_completed={len(completed)} pending={len(indexed_records)}",
        flush=True,
    )

    if workers > 1 and indexed_records:
        def run_one(item: tuple[int, dict[str, Any], str]) -> tuple[int, str, dict[str, Any]]:
            index, record, sid = item
            try:
                prediction = predict_with_retries(
                    record=record,
                    track=args.track,
                    sample_id_value=sid,
                    client=client,
                    model=args.model,
                    temperature=args.temperature,
                    max_output_tokens=args.max_output_tokens,
                    track1_image_mode=args.track1_image_mode,
                    image_root=args.image_root,
                    track2_max_input_tokens=args.track2_max_input_tokens,
                    retries=args.retries,
                )
            except Exception as exc:
                prediction = _handle_prediction_error(args, record, sid, exc)
            return index, sid, prediction

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(run_one, item) for item in indexed_records]
            for future in as_completed(futures):
                index, sid, prediction = future.result()
                append_jsonl(prediction, output_path)
                written += 1
                print(f"[{index}/{denominator}] wrote {sid}", flush=True)
                if args.sleep > 0:
                    time.sleep(args.sleep)
        print(f"Done. wrote={written} skipped={len(completed)} output={output_path}")
        return 0

    for index, record in enumerate(records, 1):
        sid = sample_id(record, args.track, index)
        if sid in completed:
            continue
        try:
            prediction = predict_with_retries(
                record=record,
                track=args.track,
                sample_id_value=sid,
                client=client,
                model=args.model,
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
                track1_image_mode=args.track1_image_mode,
                image_root=args.image_root,
                track2_max_input_tokens=args.track2_max_input_tokens,
                retries=args.retries,
            )
        except Exception as exc:
            prediction = _handle_prediction_error(args, record, sid, exc)
        append_jsonl(prediction, output_path)
        written += 1
        print(f"[{index}/{denominator}] wrote {sid}", flush=True)
        if args.sleep > 0:
            time.sleep(args.sleep)

    print(f"Done. wrote={written} skipped={len(completed)} output={output_path}")
    return 0
