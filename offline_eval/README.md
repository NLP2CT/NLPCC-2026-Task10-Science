# NLPCC 2026 Shared Task 10 Offline Evaluation Kit

This directory contains an offline evaluator for both tracks.

## Prediction Format

Gold and prediction files are JSONL. Records are matched by `id` when both files
provide unique ids; otherwise they are matched by line number.

Track 1 prediction:

```json
{"id": "track1-000001", "labels": ["Supported", "Contradiction"]}
```

Track 2 prediction:

```json
{"id": "track2-000001", "label": "Supported", "evidence_para_ids": ["P18", "P5"]}
```

## Evaluation

```bash
python offline_eval/evaluate.py --track 1 --gold gold.jsonl --pred pred.jsonl
python offline_eval/evaluate.py --track 2 --gold gold.jsonl --pred pred.jsonl
```

The command line prints a concise summary only. The full JSON report is written
next to the prediction file as `<pred_stem>_eval_result.json` by default. Use
`--output path/to/report.json` to choose another result file, or `--print-json`
to also print the full report to stdout.

Convenience wrappers:

```bash
offline_eval/eval_track1.sh gold.jsonl pred.jsonl --match id
offline_eval/eval_track2.sh gold.jsonl pred.jsonl --match id
```

## Scoring Conventions

- **Track 1 ranking score** = (Sentence Macro-F1 + Paragraph Exact Match) / 2.
- **Track 2 ranking score** = (Label Macro-F1 + Joint@3) / 2.
- When a Track 1 sentence has multiple gold labels, a prediction matching any of
  them is counted as correct. For Macro-F1, the matched gold label is used as
  the reference; if the prediction is wrong, the first gold label is used.
- The official Joint@3 metric uses `--joint-empty-policy match-empty`: when the
  gold evidence list is empty, Joint@3 succeeds only if the label is correct
  *and* the predicted evidence list is also empty. The other policies
  (`always-zero`, `exclude`) are provided for ablation only and are not used
  for ranking.

Run unit tests:

```bash
python -m unittest offline_eval.tests.test_offline_eval
```
