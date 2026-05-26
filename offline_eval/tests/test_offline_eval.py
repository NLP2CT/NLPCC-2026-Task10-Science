from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any

from sklearn.metrics import f1_score

from offline_eval.evaluate import (
    EvaluationError,
    TRACK1_LABELS,
    TRACK2_LABELS,
    evaluate_files,
    evaluate_track1,
    evaluate_track2,
    load_jsonl,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[2]
TRACK1_SOURCE = ROOT / "data" / "traindev-track-1.jsonl"
TRACK2_SOURCE = ROOT / "data" / "traindev-track-2.jsonl"

TRACK1_LINES = [1, 2, 6, 10, 20, 50, 100, 200, 500, 1000]
TRACK2_LINES = [1, 2, 376, 377, 752, 753, 1134, 1135, 1500, 2000]


def _read_selected_jsonl(path: Path, line_numbers: list[int]) -> list[dict[str, Any]]:
    wanted = set(line_numbers)
    selected: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line_no in wanted:
                selected[line_no] = json.loads(line)
            if len(selected) == len(wanted):
                break
    return [selected[line_no] for line_no in line_numbers]


def _with_ids(records: list[dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, record in enumerate(records, 1):
        item = dict(record)
        item["id"] = f"{prefix}-{index:06d}"
        result.append(item)
    return result


def _build_track1_predictions(gold_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    overrides = {
        4: {0: "Supported"},
        5: {0: "Contradiction"},
        8: {3: "Contradiction"},
        10: {0: "Supported"},
    }
    predictions: list[dict[str, Any]] = []
    for sample_index, gold in enumerate(gold_records, 1):
        labels = [sentence["types"][0] for sentence in gold["sentence_label"]]
        if sample_index == 2:
            labels[2] = "Unsupported Entity"
        if sample_index == 3:
            labels[8] = "Unsupported Causal Mechanistic"
        if sample_index == 7:
            labels[0] = "Unsupported Causal Mechanistic"
        for sent_index, label in overrides.get(sample_index, {}).items():
            labels[sent_index] = label
        predictions.append({"id": gold["id"], "labels": labels})
    return predictions


def _build_track2_predictions(gold_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    planned = [
        ("Overstate", ["P19", "P1", "P2"]),
        ("Supported", ["P125", "P4"]),
        ("Topical Match", ["P1", "P2", "P3"]),
        ("Topical Match", ["P1", "P58"]),
        ("Irrelevant", []),
        ("Topical Match", []),
        ("Supported", ["P18"]),
        ("Supported", ["P1", "P2", "P3"]),
        ("Supported", ["P10", "P10", "P1", "P8"]),
        ("Overstate", ["P1", "P2"]),
    ]
    predictions: list[dict[str, Any]] = []
    for gold, (label, evidence) in zip(gold_records, planned):
        predictions.append({"id": gold["id"], "label": label, "evidence_para_ids": evidence})
    return predictions


def _build_fixtures(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    track1_gold = _with_ids(_read_selected_jsonl(TRACK1_SOURCE, TRACK1_LINES), "track1")
    track1_pred = _build_track1_predictions(track1_gold)
    write_jsonl(track1_gold, output_dir / "track1_gold.jsonl")
    write_jsonl(track1_pred, output_dir / "track1_pred.jsonl")

    track2_gold = _with_ids(_read_selected_jsonl(TRACK2_SOURCE, TRACK2_LINES), "track2")
    track2_pred = _build_track2_predictions(track2_gold)
    write_jsonl(track2_gold, output_dir / "track2_gold.jsonl")
    write_jsonl(track2_pred, output_dir / "track2_pred.jsonl")
    return output_dir


class OfflineEvalTest(unittest.TestCase):
    def assert_close(self, actual: float, expected: float, places: int = 10) -> None:
        self.assertAlmostEqual(actual, expected, places=places)

    def make_examples(self, tmpdir: str) -> Path:
        return _build_fixtures(Path(tmpdir) / "examples")

    def test_generated_track1_fixture_matches_expected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = self.make_examples(tmpdir)
            report = evaluate_files(
                1,
                out / "track1_gold.jsonl",
                out / "track1_pred.jsonl",
                match="id",
            )
            self.assert_close(report["macro_f1"], 0.8008774746, places=10)
            self.assert_close(report["pem"], 0.6)
            self.assert_close(report["score"], 0.7004387373, places=10)

            sklearn_f1 = f1_score(
                report["y_true"],
                report["y_pred"],
                labels=TRACK1_LABELS,
                average="macro",
                zero_division=0,
            )
            self.assert_close(report["macro_f1"], sklearn_f1)

    def test_generated_track2_fixture_matches_expected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = self.make_examples(tmpdir)
            report = evaluate_files(
                2,
                out / "track2_gold.jsonl",
                out / "track2_pred.jsonl",
                match="id",
            )
            self.assert_close(report["macro_f1"], 0.6791666667, places=10)
            self.assert_close(report["joint_at_3"], 0.5)
            self.assert_close(report["score"], 0.5895833333, places=10)

            sklearn_f1 = f1_score(
                report["y_true"],
                report["y_pred"],
                labels=TRACK2_LABELS,
                average="macro",
                zero_division=0,
            )
            self.assert_close(report["macro_f1"], sklearn_f1)

    def test_id_matching_allows_shuffled_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = self.make_examples(tmpdir)
            gold = load_jsonl(out / "track2_gold.jsonl")
            pred = load_jsonl(out / "track2_pred.jsonl")
            shuffled = list(reversed(pred))
            report = evaluate_track2(gold, shuffled, match="id")
            self.assert_close(report["score"], 0.5895833333333333)

    def test_joint_empty_policies(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = self.make_examples(tmpdir)
            gold = load_jsonl(out / "track2_gold.jsonl")
            pred = load_jsonl(out / "track2_pred.jsonl")
            match_empty = evaluate_track2(gold, pred, match="id", joint_empty_policy="match-empty")
            always_zero = evaluate_track2(gold, pred, match="id", joint_empty_policy="always-zero")
            exclude = evaluate_track2(gold, pred, match="id", joint_empty_policy="exclude")
            self.assert_close(match_empty["joint_at_3"], 0.5)
            self.assert_close(always_zero["joint_at_3"], 0.4)
            self.assert_close(exclude["joint_at_3"], 4 / 8)
            self.assertEqual(exclude["joint_at_3_denominator"], 8)

    def test_invalid_label_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = self.make_examples(tmpdir)
            gold = load_jsonl(out / "track2_gold.jsonl")
            pred = load_jsonl(out / "track2_pred.jsonl")
            pred[0]["label"] = "Not A Label"
            with self.assertRaises(EvaluationError):
                evaluate_track2(gold, pred, match="id")

    def test_duplicate_id_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = self.make_examples(tmpdir)
            gold = load_jsonl(out / "track1_gold.jsonl")
            pred = load_jsonl(out / "track1_pred.jsonl")
            pred[1]["id"] = pred[0]["id"]
            with self.assertRaises(EvaluationError):
                evaluate_track1(gold, pred, match="id")

    def test_id_set_mismatch_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = self.make_examples(tmpdir)
            gold = load_jsonl(out / "track1_gold.jsonl")
            pred = load_jsonl(out / "track1_pred.jsonl")
            pred[0]["id"] = "unknown-id"
            with self.assertRaises(EvaluationError):
                evaluate_track1(gold, pred, match="id")

    def test_line_count_mismatch_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = self.make_examples(tmpdir)
            gold = load_jsonl(out / "track1_gold.jsonl")
            pred = load_jsonl(out / "track1_pred.jsonl")[:-1]
            with self.assertRaises(EvaluationError):
                evaluate_track1(gold, pred, match="line")

    def test_track1_label_count_mismatch_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = self.make_examples(tmpdir)
            gold = load_jsonl(out / "track1_gold.jsonl")
            pred = load_jsonl(out / "track1_pred.jsonl")
            pred[0]["labels"] = pred[0]["labels"][:-1]
            with self.assertRaises(EvaluationError):
                evaluate_track1(gold, pred, match="id")

    def test_cli_file_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = self.make_examples(tmpdir)
            # Re-write one tiny file pair to exercise JSONL helpers directly.
            gold = load_jsonl(out / "track1_gold.jsonl")[:1]
            pred = load_jsonl(out / "track1_pred.jsonl")[:1]
            gold_path = Path(tmpdir) / "tiny_gold.jsonl"
            pred_path = Path(tmpdir) / "tiny_pred.jsonl"
            write_jsonl(gold, gold_path)
            write_jsonl(pred, pred_path)
            report = evaluate_files(1, gold_path, pred_path, match="id")
            self.assertEqual(report["num_samples"], 1)
            self.assert_close(report["pem"], 1.0)

    def test_cli_prints_summary_and_writes_result_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = self.make_examples(tmpdir)
            pred_path = out / "track2_pred.jsonl"
            result_path = out / "track2_pred_eval_result.json"
            completed = subprocess.run(
                [
                    "python",
                    str(ROOT / "offline_eval" / "evaluate.py"),
                    "--track",
                    "2",
                    "--gold",
                    str(out / "track2_gold.jsonl"),
                    "--pred",
                    str(pred_path),
                    "--match",
                    "id",
                ],
                check=True,
                text=True,
                capture_output=True,
            )
            self.assertIn("Track 2 evaluation complete", completed.stdout)
            self.assertIn("score: 0.5895833333", completed.stdout)
            self.assertNotIn('"samples"', completed.stdout)
            self.assertTrue(result_path.exists())
            report = json.loads(result_path.read_text(encoding="utf-8"))
            self.assert_close(report["score"], 0.5895833333333333)


if __name__ == "__main__":
    unittest.main()
