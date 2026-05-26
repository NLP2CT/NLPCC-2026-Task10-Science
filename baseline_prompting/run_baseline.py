#!/usr/bin/env python3
"""CLI entrypoint for NLPCC 2026 Shared Task 10 prompting baselines."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baseline_prompting.client import OpenAICompatibleClient, normalize_base_url
from baseline_prompting.errors import BaselineError
from baseline_prompting.io_utils import append_jsonl, load_jsonl, read_completed_ids, sample_id
from baseline_prompting.runner import predict_record, run_baseline, write_error
from baseline_prompting.tasks import (
    build_track1_messages,
    build_track2_messages,
    evidence_caption,
    extract_json_object,
    fallback_prediction,
    image_data_url,
    iter_track2_paragraphs,
    parse_track1_response,
    parse_track2_response,
    resolve_image_path,
    track1_sentences,
    truncate_track2_paragraphs_by_tokens,
    validate_label,
)


def env_default(name: str, fallback: str = "") -> str:
    return os.environ.get(name, fallback)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run single-turn prompting baselines.")
    parser.add_argument("--track", type=int, choices=(1, 2), required=True)
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Prediction JSONL output file")
    parser.add_argument("--base-url", default=env_default("OPENAI_BASE_URL"))
    parser.add_argument("--api-key", default=env_default("OPENAI_API_KEY"))
    parser.add_argument("--model", default=env_default("OPENAI_MODEL", "gpt-4o-mini"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1, help="Number of concurrent API requests")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between samples")
    parser.add_argument("--limit", type=int, help="Only process the first N input records")
    parser.add_argument("--resume", action="store_true", help="Append only missing ids when output already exists")
    parser.add_argument("--on-error", choices=("fallback", "raise"), default="fallback")
    parser.add_argument("--error-log", help="JSONL error log path")
    parser.add_argument("--track1-image-mode", choices=("data-url", "text"), default="data-url")
    parser.add_argument("--image-root", default="data", help="Root directory for relative Track 1 img_path values")
    parser.add_argument("--track2-max-input-tokens", type=int, default=64000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        return run_baseline(args)
    except BaselineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


__all__ = [
    "BaselineError",
    "OpenAICompatibleClient",
    "append_jsonl",
    "build_arg_parser",
    "build_track1_messages",
    "build_track2_messages",
    "evidence_caption",
    "extract_json_object",
    "fallback_prediction",
    "image_data_url",
    "iter_track2_paragraphs",
    "load_jsonl",
    "main",
    "normalize_base_url",
    "parse_track1_response",
    "parse_track2_response",
    "predict_record",
    "read_completed_ids",
    "resolve_image_path",
    "run_baseline",
    "sample_id",
    "track1_sentences",
    "truncate_track2_paragraphs_by_tokens",
    "validate_label",
    "write_error",
]


if __name__ == "__main__":
    raise SystemExit(main())
