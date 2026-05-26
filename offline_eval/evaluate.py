#!/usr/bin/env python3
"""Offline evaluator for NLPCC 2026 Shared Task 10.

The evaluator accepts JSONL gold and prediction files for either track. Records
are matched by id when possible, otherwise by line number.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable


TRACK1_LABELS = [
    "Supported",
    "Unsupported Causal Mechanistic",
    "Unsupported Entity",
    "Scope Overgeneralization",
    "Contradiction",
]

TRACK2_LABELS = [
    "Supported",
    "Overstate",
    "Topical Match",
    "Irrelevant",
]

JOINT_EMPTY_POLICIES = ("match-empty", "always-zero", "exclude")


class EvaluationError(ValueError):
    """Raised for invalid input files or malformed predictions."""


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvaluationError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(obj, dict):
                raise EvaluationError(f"{path}:{line_no}: each JSONL line must be an object")
            records.append(obj)
    return records


def write_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def has_unique_ids(records: list[dict[str, Any]]) -> bool:
    ids = [record.get("id") for record in records]
    return all(isinstance(value, str) and value for value in ids) and len(set(ids)) == len(ids)


def resolve_pairs(
    gold_records: list[dict[str, Any]],
    pred_records: list[dict[str, Any]],
    match: str,
    track: int,
) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    if match not in {"auto", "id", "line"}:
        raise EvaluationError(f"unsupported match mode: {match}")

    if match == "auto":
        match = "id" if has_unique_ids(gold_records) and has_unique_ids(pred_records) else "line"

    if match == "line":
        if len(gold_records) != len(pred_records):
            raise EvaluationError(
                f"line matching requires the same number of records: "
                f"gold={len(gold_records)} pred={len(pred_records)}"
            )
        pairs = []
        for index, (gold, pred) in enumerate(zip(gold_records, pred_records), 1):
            sample_id = gold.get("id") or pred.get("id") or f"track{track}-{index:06d}"
            if not isinstance(sample_id, str):
                sample_id = f"track{track}-{index:06d}"
            pairs.append((sample_id, gold, pred))
        return pairs

    if not has_unique_ids(gold_records):
        raise EvaluationError("id matching requires every gold record to have a unique string id")
    if not has_unique_ids(pred_records):
        raise EvaluationError("id matching requires every prediction record to have a unique string id")

    gold_by_id = {record["id"]: record for record in gold_records}
    pred_by_id = {record["id"]: record for record in pred_records}
    gold_ids = set(gold_by_id)
    pred_ids = set(pred_by_id)
    if gold_ids != pred_ids:
        missing = sorted(gold_ids - pred_ids)[:10]
        extra = sorted(pred_ids - gold_ids)[:10]
        details = []
        if missing:
            details.append(f"missing predictions for ids={missing}")
        if extra:
            details.append(f"extra prediction ids={extra}")
        raise EvaluationError("gold/pred id sets differ: " + "; ".join(details))

    return [(sample_id, gold_by_id[sample_id], pred_by_id[sample_id]) for sample_id in gold_by_id]


def validate_label(label: Any, allowed: list[str], field: str, sample_id: str) -> str:
    if not isinstance(label, str):
        raise EvaluationError(f"{sample_id}: {field} must be a string")
    if label not in allowed:
        raise EvaluationError(f"{sample_id}: invalid {field}={label!r}; allowed={allowed}")
    return label


def extract_track1_pred_labels(pred: dict[str, Any], sample_id: str) -> list[str]:
    if "labels" in pred:
        labels = pred["labels"]
        if not isinstance(labels, list):
            raise EvaluationError(f"{sample_id}: labels must be a list")
        return [validate_label(label, TRACK1_LABELS, "label", sample_id) for label in labels]

    # Compatibility path for predictions mirroring sentence_label-like data.
    if "sentence_label" in pred:
        sentence_label = pred["sentence_label"]
        if not isinstance(sentence_label, list):
            raise EvaluationError(f"{sample_id}: sentence_label must be a list")
        labels: list[str] = []
        for index, item in enumerate(sentence_label, 1):
            if not isinstance(item, dict):
                raise EvaluationError(f"{sample_id}: sentence_label[{index}] must be an object")
            if "label" in item:
                label = item["label"]
            elif "type" in item:
                label = item["type"]
            elif "types" in item and isinstance(item["types"], list) and item["types"]:
                label = item["types"][0]
            else:
                raise EvaluationError(
                    f"{sample_id}: sentence_label[{index}] must contain label, type, or non-empty types"
                )
            labels.append(validate_label(label, TRACK1_LABELS, "label", sample_id))
        return labels

    raise EvaluationError(f"{sample_id}: Track 1 prediction must contain labels")


def extract_track1_gold_labels(gold: dict[str, Any], sample_id: str) -> list[list[str]]:
    sentence_label = gold.get("sentence_label")
    if not isinstance(sentence_label, list):
        raise EvaluationError(f"{sample_id}: gold sentence_label must be a list")
    labels: list[list[str]] = []
    for index, item in enumerate(sentence_label, 1):
        if not isinstance(item, dict) or not isinstance(item.get("types"), list) or not item["types"]:
            raise EvaluationError(f"{sample_id}: gold sentence_label[{index}] must contain non-empty types")
        gold_labels = [validate_label(label, TRACK1_LABELS, "gold label", sample_id) for label in item["types"]]
        labels.append(gold_labels)
    return labels


def precision_recall_f1(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def classification_report(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
) -> tuple[float, dict[str, dict[str, float | int]]]:
    per_label: dict[str, dict[str, float | int]] = {}
    for label in labels:
        tp = sum(1 for truth, pred in zip(y_true, y_pred) if truth == label and pred == label)
        fp = sum(1 for truth, pred in zip(y_true, y_pred) if truth != label and pred == label)
        fn = sum(1 for truth, pred in zip(y_true, y_pred) if truth == label and pred != label)
        per_label[label] = precision_recall_f1(tp, fp, fn)
        per_label[label]["support"] = sum(1 for truth in y_true if truth == label)
        per_label[label]["predicted"] = sum(1 for pred in y_pred if pred == label)
    macro_f1 = sum(float(values["f1"]) for values in per_label.values()) / len(labels)
    return macro_f1, per_label


def evaluate_track1(
    gold_records: list[dict[str, Any]],
    pred_records: list[dict[str, Any]],
    match: str = "auto",
) -> dict[str, Any]:
    pairs = resolve_pairs(gold_records, pred_records, match, track=1)
    y_true: list[str] = []
    y_pred: list[str] = []
    sentence_correct = 0
    paragraph_correct = 0
    sample_details: list[dict[str, Any]] = []

    for sample_id, gold, pred in pairs:
        gold_labels = extract_track1_gold_labels(gold, sample_id)
        pred_labels = extract_track1_pred_labels(pred, sample_id)
        if len(gold_labels) != len(pred_labels):
            raise EvaluationError(
                f"{sample_id}: Track 1 label count mismatch: "
                f"gold={len(gold_labels)} pred={len(pred_labels)}"
            )

        all_correct = True
        correct_count = 0
        normalized_true: list[str] = []
        for index, (gold_options, pred_label) in enumerate(zip(gold_labels, pred_labels), 1):
            is_correct = pred_label in gold_options
            if is_correct:
                true_label = pred_label
                correct_count += 1
            else:
                true_label = gold_options[0]
                all_correct = False
            y_true.append(true_label)
            y_pred.append(pred_label)
            normalized_true.append(true_label)

        sentence_correct += correct_count
        paragraph_correct += 1 if all_correct else 0
        sample_details.append(
            {
                "id": sample_id,
                "num_sentences": len(gold_labels),
                "num_correct": correct_count,
                "paragraph_exact_match": all_correct,
                "gold_normalized": normalized_true,
                "predicted": pred_labels,
            }
        )

    macro_f1, per_label = classification_report(y_true, y_pred, TRACK1_LABELS)
    pem = paragraph_correct / len(pairs) if pairs else 0.0
    score = (macro_f1 + pem) / 2
    return {
        "track": 1,
        "num_samples": len(pairs),
        "num_sentences": len(y_true),
        "sentence_accuracy": sentence_correct / len(y_true) if y_true else 0.0,
        "macro_f1": macro_f1,
        "pem": pem,
        "score": score,
        "labels": TRACK1_LABELS,
        "per_label": per_label,
        "y_true": y_true,
        "y_pred": y_pred,
        "samples": sample_details,
    }


def dedupe_preserve_order(values: Any, sample_id: str) -> list[str]:
    if values is None:
        return []
    if not isinstance(values, list):
        raise EvaluationError(f"{sample_id}: evidence_para_ids must be a list")
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not isinstance(value, str):
            raise EvaluationError(f"{sample_id}: evidence_para_ids must contain strings")
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def extract_track2_gold(gold: dict[str, Any], sample_id: str) -> tuple[str, list[str]]:
    label = validate_label(gold.get("label"), TRACK2_LABELS, "gold label", sample_id)
    evidence = dedupe_preserve_order(gold.get("evidence_para_ids"), sample_id)
    return label, evidence


def extract_track2_pred(pred: dict[str, Any], sample_id: str) -> tuple[str, list[str]]:
    label = validate_label(pred.get("label"), TRACK2_LABELS, "label", sample_id)
    evidence = dedupe_preserve_order(pred.get("evidence_para_ids", []), sample_id)
    return label, evidence[:3]


def joint_success(
    gold_label: str,
    gold_evidence: list[str],
    pred_label: str,
    pred_evidence_top3: list[str],
    empty_policy: str,
) -> bool | None:
    if empty_policy not in JOINT_EMPTY_POLICIES:
        raise EvaluationError(f"unsupported empty evidence policy: {empty_policy}")
    if not gold_evidence:
        if empty_policy == "exclude":
            return None
        if empty_policy == "always-zero":
            return False
        return gold_label == pred_label and not pred_evidence_top3
    return gold_label == pred_label and bool(set(gold_evidence).intersection(pred_evidence_top3))


def evaluate_track2(
    gold_records: list[dict[str, Any]],
    pred_records: list[dict[str, Any]],
    match: str = "auto",
    joint_empty_policy: str = "match-empty",
) -> dict[str, Any]:
    pairs = resolve_pairs(gold_records, pred_records, match, track=2)
    y_true: list[str] = []
    y_pred: list[str] = []
    joint_values: list[int] = []
    sample_details: list[dict[str, Any]] = []
    empty_gold_count = 0

    for sample_id, gold, pred in pairs:
        gold_label, gold_evidence = extract_track2_gold(gold, sample_id)
        pred_label, pred_evidence = extract_track2_pred(pred, sample_id)
        y_true.append(gold_label)
        y_pred.append(pred_label)

        if not gold_evidence:
            empty_gold_count += 1
        joint = joint_success(
            gold_label,
            gold_evidence,
            pred_label,
            pred_evidence,
            empty_policy=joint_empty_policy,
        )
        if joint is not None:
            joint_values.append(1 if joint else 0)
        sample_details.append(
            {
                "id": sample_id,
                "gold_label": gold_label,
                "pred_label": pred_label,
                "gold_evidence_para_ids": gold_evidence,
                "pred_evidence_para_ids_top3": pred_evidence,
                "label_correct": gold_label == pred_label,
                "joint_at_3_success": joint,
            }
        )

    macro_f1, per_label = classification_report(y_true, y_pred, TRACK2_LABELS)
    joint_at_3 = sum(joint_values) / len(joint_values) if joint_values else 0.0
    score = (macro_f1 + joint_at_3) / 2
    return {
        "track": 2,
        "num_samples": len(pairs),
        "macro_f1": macro_f1,
        "joint_at_3": joint_at_3,
        "joint_at_3_denominator": len(joint_values),
        "empty_gold_evidence_count": empty_gold_count,
        "joint_empty_policy": joint_empty_policy,
        "score": score,
        "labels": TRACK2_LABELS,
        "per_label": per_label,
        "y_true": y_true,
        "y_pred": y_pred,
        "samples": sample_details,
    }


def evaluate_files(
    track: int,
    gold_path: str | Path,
    pred_path: str | Path,
    match: str = "auto",
    joint_empty_policy: str = "match-empty",
) -> dict[str, Any]:
    gold_records = load_jsonl(gold_path)
    pred_records = load_jsonl(pred_path)
    if track == 1:
        return evaluate_track1(gold_records, pred_records, match=match)
    if track == 2:
        return evaluate_track2(
            gold_records,
            pred_records,
            match=match,
            joint_empty_policy=joint_empty_policy,
        )
    raise EvaluationError(f"unsupported track: {track}")


def default_output_path(pred_path: str | Path) -> Path:
    path = Path(pred_path)
    if path.suffix:
        return path.with_name(f"{path.stem}_eval_result.json")
    return path.with_name(f"{path.name}_eval_result.json")


def summary_lines(report: dict[str, Any], output_path: str | Path) -> list[str]:
    lines = [
        f"Track {report['track']} evaluation complete",
        f"result_file: {output_path}",
        f"score: {report['score']:.10f}",
        f"macro_f1: {report['macro_f1']:.10f}",
        f"num_samples: {report['num_samples']}",
    ]
    if report["track"] == 1:
        lines.extend(
            [
                f"pem: {report['pem']:.10f}",
                f"num_sentences: {report['num_sentences']}",
                f"sentence_accuracy: {report['sentence_accuracy']:.10f}",
            ]
        )
    else:
        lines.extend(
            [
                f"joint_at_3: {report['joint_at_3']:.10f}",
                f"joint_at_3_denominator: {report['joint_at_3_denominator']}",
                f"empty_gold_evidence_count: {report['empty_gold_evidence_count']}",
                f"joint_empty_policy: {report['joint_empty_policy']}",
            ]
        )
    return lines


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate NLPCC 2026 Shared Task 10 predictions.")
    parser.add_argument("--track", type=int, choices=(1, 2), required=True)
    parser.add_argument("--gold", required=True, help="Gold JSONL file")
    parser.add_argument("--pred", required=True, help="Prediction JSONL file")
    parser.add_argument("--match", choices=("auto", "id", "line"), default="auto")
    parser.add_argument("--joint-empty-policy", choices=JOINT_EMPTY_POLICIES, default="match-empty")
    parser.add_argument(
        "--output",
        help="JSON report output path. Defaults to <pred_stem>_eval_result.json next to the prediction file.",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Write compact JSON to the result file instead of pretty-printed JSON.",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Also print the full JSON report to stdout. By default only a concise summary is printed.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        report = evaluate_files(
            track=args.track,
            gold_path=args.gold,
            pred_path=args.pred,
            match=args.match,
            joint_empty_policy=args.joint_empty_policy,
        )
    except EvaluationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    output_path = Path(args.output) if args.output else default_output_path(args.pred)
    indent = None if args.compact else 2
    text = json.dumps(report, ensure_ascii=False, indent=indent, sort_keys=False)
    output_path.write_text(text + "\n", encoding="utf-8")

    print("\n".join(summary_lines(report, output_path)))
    if args.print_json:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
