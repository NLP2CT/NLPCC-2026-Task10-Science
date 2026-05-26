"""JSONL and id helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from baseline_prompting.errors import BaselineError


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
                raise BaselineError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(obj, dict):
                raise BaselineError(f"{path}:{line_no}: each JSONL line must be an object")
            records.append(obj)
    return records


def append_jsonl(record: dict[str, Any], path: str | Path) -> None:
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def sample_id(record: dict[str, Any], track: int, index: int) -> str:
    value = record.get("id")
    if isinstance(value, str) and value:
        return value
    return f"track{track}-{index:06d}"


def read_completed_ids(path: str | Path) -> set[str]:
    output = Path(path)
    if not output.exists():
        return set()
    completed: set[str] = set()
    for record in load_jsonl(output):
        value = record.get("id")
        if isinstance(value, str) and value:
            completed.add(value)
    return completed

