from __future__ import annotations

import argparse
import contextlib
import io
import json
import threading
import time
import tempfile
import unittest
from pathlib import Path

from baseline_prompting import prepare_dev_eval
from baseline_prompting import run_baseline


class StaticClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0
        self.messages = []

    def chat_completion(self, messages, model, temperature, max_output_tokens):
        self.calls += 1
        self.messages.append(messages)
        return self.response


class SlowThreadClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def chat_completion(self, messages, model, temperature, max_output_tokens):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.05)
        with self.lock:
            self.active -= 1
        return self.response


def write_jsonl(records, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


class BaselinePromptingTests(unittest.TestCase):
    def track1_record(self):
        return {
            "id": "t1",
            "claim_text": "A. B.",
            "evidence_bundle": [
                {
                    "type": "image",
                    "image_caption": ["Figure 1: result."],
                    "img_path": "images/test.jpg",
                }
            ],
            "sentence_label": [{"sentence": "A."}, {"sentence": "B."}],
        }

    def test_track1_default_multimodal_message_contains_data_url(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            image = root / "images" / "test.jpg"
            image.parent.mkdir()
            image.write_bytes(b"fake-jpeg")
            messages = run_baseline.build_track1_messages(self.track1_record(), "data-url", root)
        content = messages[1]["content"]
        self.assertTrue(any(part["type"] == "image_url" for part in content))
        image_part = next(part for part in content if part["type"] == "image_url")
        self.assertTrue(image_part["image_url"]["url"].startswith("data:image/jpeg;base64,"))

    def test_track1_text_mode_has_no_image_part(self):
        messages = run_baseline.build_track1_messages(self.track1_record(), "text", "missing-root")
        self.assertFalse(any(part["type"] == "image_url" for part in messages[1]["content"]))

    def test_track1_parse_validates_label_count_and_labels(self):
        labels = run_baseline.parse_track1_response('```json\n{"labels":["Supported","Contradiction"]}\n```', 2)
        self.assertEqual(labels, ["Supported", "Contradiction"])
        with self.assertRaises(run_baseline.BaselineError):
            run_baseline.parse_track1_response('{"labels":["Supported"]}', 2)
        with self.assertRaises(run_baseline.BaselineError):
            run_baseline.parse_track1_response('{"labels":["Bad","Supported"]}', 2)

    def test_track2_truncates_on_paragraph_boundaries_and_is_text_only(self):
        record = {
            "id": "t2",
            "claim_text": "Claim.",
            "cited_paper_full_text": [
                {"P1": "a" * 20},
                {"P2": "b" * 20},
                {"P3": "c" * 20},
            ],
        }
        messages, allowed = run_baseline.build_track2_messages(record, max_input_tokens=7)
        self.assertEqual(allowed, {"P1"})
        self.assertIsInstance(messages[1]["content"], str)
        self.assertIn("Only the included prefix", messages[1]["content"])
        self.assertIn("P1:", messages[1]["content"])
        self.assertNotIn("P2:", messages[1]["content"])

    def test_track2_parse_dedupes_top3_and_rejects_unknown_ids(self):
        label, evidence = run_baseline.parse_track2_response(
            '{"label":"Overstate","evidence_para_ids":["P2","P2","P1","P3","P4"]}',
            {"P1", "P2", "P3", "P4"},
        )
        self.assertEqual(label, "Overstate")
        self.assertEqual(evidence, ["P2", "P1", "P3"])
        with self.assertRaises(run_baseline.BaselineError):
            run_baseline.parse_track2_response('{"label":"Supported","evidence_para_ids":["P9"]}', {"P1"})

    def test_prepare_dev_eval_generates_input_and_gold(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "source.jsonl"
            out_input = root / "input.jsonl"
            out_gold = root / "gold.jsonl"
            write_jsonl(
                [
                    {
                        "claim_text": "A.",
                        "evidence_bundle": [],
                        "sentence_label": [{"sentence": "A.", "types": ["Supported"]}],
                    }
                ],
                source,
            )
            records = prepare_dev_eval.load_jsonl(source)
            inputs, gold = prepare_dev_eval.prepare(records, track=1, limit=None)
            prepare_dev_eval.write_jsonl(inputs, out_input)
            prepare_dev_eval.write_jsonl(gold, out_gold)
            input_record = prepare_dev_eval.load_jsonl(out_input)[0]
            gold_record = prepare_dev_eval.load_jsonl(out_gold)[0]
        self.assertEqual(input_record["id"], "track1-000001")
        self.assertNotIn("types", input_record["sentence_label"][0])
        self.assertEqual(gold_record["sentence_label"][0]["types"], ["Supported"])

    def test_run_baseline_resume_skips_existing_prediction(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            input_path = root / "input.jsonl"
            output_path = root / "pred.jsonl"
            write_jsonl([self.track1_record()], input_path)
            write_jsonl([{"id": "t1", "labels": ["Supported", "Supported"]}], output_path)
            client = StaticClient('{"labels":["Contradiction","Contradiction"]}')
            args = argparse.Namespace(
                track=1,
                input=str(input_path),
                output=str(output_path),
                resume=True,
                mock=False,
                base_url="",
                api_key="",
                timeout=1,
                limit=None,
                model="mock",
                temperature=0,
                max_output_tokens=100,
                track1_image_mode="text",
                image_root=str(root),
                track2_max_input_tokens=1000,
                error_log=None,
                on_error="raise",
                sleep=0,
                retries=0,
                workers=1,
            )
            run_baseline.run_baseline(args, client=client)
            rows = run_baseline.load_jsonl(output_path)
        self.assertEqual(client.calls, 0)
        self.assertEqual(len(rows), 1)

    def test_fallback_on_error_writes_supported_prediction_and_error_log(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            input_path = root / "input.jsonl"
            output_path = root / "pred.jsonl"
            error_path = root / "errors.jsonl"
            write_jsonl([self.track1_record()], input_path)
            client = StaticClient('{"labels":["Bad","Bad"]}')
            args = argparse.Namespace(
                track=1,
                input=str(input_path),
                output=str(output_path),
                resume=False,
                mock=False,
                base_url="",
                api_key="",
                timeout=1,
                limit=None,
                model="mock",
                temperature=0,
                max_output_tokens=100,
                track1_image_mode="text",
                image_root=str(root),
                track2_max_input_tokens=1000,
                error_log=str(error_path),
                on_error="fallback",
                sleep=0,
                retries=0,
                workers=1,
            )
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                run_baseline.run_baseline(args, client=client)
            rows = run_baseline.load_jsonl(output_path)
            errors = run_baseline.load_jsonl(error_path)
        self.assertEqual(rows[0]["labels"], ["Supported", "Supported"])
        self.assertEqual(errors[0]["id"], "t1")
        self.assertIn("[error] t1:", stderr.getvalue())
        self.assertIn("invalid Track 1 label", stderr.getvalue())

    def test_run_baseline_workers_process_records_concurrently(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            input_path = root / "input.jsonl"
            output_path = root / "pred.jsonl"
            records = []
            for index in range(4):
                record = self.track1_record()
                record["id"] = f"t{index}"
                records.append(record)
            write_jsonl(records, input_path)
            client = SlowThreadClient('{"labels":["Supported","Supported"]}')
            args = argparse.Namespace(
                track=1,
                input=str(input_path),
                output=str(output_path),
                resume=False,
                mock=False,
                base_url="",
                api_key="",
                timeout=1,
                limit=None,
                model="mock",
                temperature=0,
                max_output_tokens=100,
                track1_image_mode="text",
                image_root=str(root),
                track2_max_input_tokens=1000,
                error_log=None,
                on_error="raise",
                sleep=0,
                retries=0,
                workers=4,
            )
            run_baseline.run_baseline(args, client=client)
            rows = run_baseline.load_jsonl(output_path)
        self.assertEqual(len(rows), 4)
        self.assertGreater(client.max_active, 1)

    def test_resume_retries_ids_with_existing_error_log(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            input_path = root / "input.jsonl"
            output_path = root / "pred.jsonl"
            error_path = root / "errors.jsonl"
            write_jsonl([self.track1_record()], input_path)
            write_jsonl([{"id": "t1", "labels": ["Supported", "Supported"]}], output_path)
            write_jsonl([{"id": "t1", "error": "old failure"}], error_path)
            client = StaticClient('{"labels":["Contradiction","Contradiction"]}')
            args = argparse.Namespace(
                track=1,
                input=str(input_path),
                output=str(output_path),
                resume=True,
                mock=False,
                base_url="",
                api_key="",
                timeout=1,
                limit=None,
                model="mock",
                temperature=0,
                max_output_tokens=100,
                track1_image_mode="text",
                image_root=str(root),
                track2_max_input_tokens=1000,
                error_log=str(error_path),
                on_error="raise",
                sleep=0,
                retries=0,
                workers=1,
            )
            run_baseline.run_baseline(args, client=client)
            rows = run_baseline.load_jsonl(output_path)
        self.assertEqual(client.calls, 1)
        self.assertEqual(rows[-1]["labels"], ["Contradiction", "Contradiction"])

    def test_resume_prints_completed_retry_and_pending_counts(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            input_path = root / "input.jsonl"
            output_path = root / "pred.jsonl"
            error_path = root / "errors.jsonl"
            records = []
            for index in range(3):
                record = self.track1_record()
                record["id"] = f"t{index}"
                records.append(record)
            write_jsonl(records, input_path)
            write_jsonl(
                [
                    {"id": "t0", "labels": ["Supported", "Supported"]},
                    {"id": "t1", "labels": ["Supported", "Supported"]},
                ],
                output_path,
            )
            write_jsonl([{"id": "t1", "error": "old failure"}], error_path)
            client = StaticClient('{"labels":["Supported","Supported"]}')
            args = argparse.Namespace(
                track=1,
                input=str(input_path),
                output=str(output_path),
                resume=True,
                mock=False,
                base_url="",
                api_key="",
                timeout=1,
                limit=None,
                model="mock",
                temperature=0,
                max_output_tokens=100,
                track1_image_mode="text",
                image_root=str(root),
                track2_max_input_tokens=1000,
                error_log=str(error_path),
                on_error="raise",
                sleep=0,
                retries=0,
                workers=1,
            )
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                run_baseline.run_baseline(args, client=client)
        text = stdout.getvalue()
        self.assertIn("Resume: existing_predictions=2 retry_errors=1 skipped_completed=1 pending=2", text)


if __name__ == "__main__":
    unittest.main()
