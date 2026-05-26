#!/usr/bin/env python3
"""Prepare test-like dev inputs and gold files from train-dev JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


class PrepareError(ValueError):
    """Raised for invalid source data."""


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
                raise PrepareError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(obj, dict):
                raise PrepareError(f"{path}:{line_no}: each JSONL line must be an object")
            records.append(obj)
    return records


def write_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def with_id(record: dict[str, Any], track: int, index: int) -> dict[str, Any]:
    result = dict(record)
    result["id"] = record.get("id") if isinstance(record.get("id"), str) and record.get("id") else f"track{track}-{index:06d}"
    return result


def strip_track1(record: dict[str, Any]) -> dict[str, Any]:
    stripped = {
        "id": record["id"],
        "claim_text": record.get("claim_text", ""),
        "evidence_bundle": record.get("evidence_bundle", []),
        "sentence_label": [],
    }
    sentence_label = record.get("sentence_label")
    if not isinstance(sentence_label, list):
        raise PrepareError(f"{record['id']}: Track 1 source must contain sentence_label list")
    for item in sentence_label:
        if not isinstance(item, dict) or not isinstance(item.get("sentence"), str):
            raise PrepareError(f"{record['id']}: Track 1 sentence_label item must contain sentence")
        stripped["sentence_label"].append({"sentence": item["sentence"]})
    return stripped


def strip_track2(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record["id"],
        "claim_text": record.get("claim_text", ""),
        "cited_paper_full_text": record.get("cited_paper_full_text", []),
    }


def prepare(records: list[dict[str, Any]], track: int, limit: int | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if limit is not None:
        records = records[:limit]
    gold: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    for index, record in enumerate(records, 1):
        gold_record = with_id(record, track, index)
        gold.append(gold_record)
        inputs.append(strip_track1(gold_record) if track == 1 else strip_track2(gold_record))
    return inputs, gold


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare baseline dev input/gold JSONL files.")
    parser.add_argument("--track", type=int, choices=(1, 2), required=True)
    parser.add_argument("--source", required=True, help="Train-dev source JSONL")
    parser.add_argument("--output-input", required=True, help="Test-like input JSONL")
    parser.add_argument("--output-gold", required=True, help="Gold JSONL with ids")
    parser.add_argument("--limit", type=int, help="Only use the first N records")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    records = load_jsonl(args.source)
    inputs, gold = prepare(records, args.track, args.limit)
    write_jsonl(inputs, args.output_input)
    write_jsonl(gold, args.output_gold)
    print(f"Prepared {len(inputs)} Track {args.track} records")
    print(f"input: {args.output_input}")
    print(f"gold: {args.output_gold}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

