# Baseline Prompting Kit

This directory contains simple single-turn prompting baselines for NLPCC 2026
Shared Task 10.

Prompt wording lives in `baseline_prompting/prompts/`.

## Prepare Dev Data

```bash
python baseline_prompting/prepare_dev_eval.py --track 1 \
  --source data/traindev-track-1.jsonl \
  --output-input outputs/track1_dev_input.jsonl \
  --output-gold outputs/track1_dev_gold.jsonl \
  --limit 100

python baseline_prompting/prepare_dev_eval.py --track 2 \
  --source data/traindev-track-2.jsonl \
  --output-input outputs/track2_dev_input.jsonl \
  --output-gold outputs/track2_dev_gold.jsonl \
  --limit 100
```

## Run

```bash
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_API_KEY="..."
export OPENAI_MODEL="gpt-4o-mini"

python baseline_prompting/run_baseline.py \
  --track 1 \
  --input outputs/track1_dev_input.jsonl \
  --output outputs/track1_dev_pred.jsonl

python baseline_prompting/run_baseline.py \
  --track 2 \
  --input outputs/track2_dev_input.jsonl \
  --output outputs/track2_dev_pred.jsonl
```

Use `--workers N` to run concurrent API requests. Start conservatively, e.g.
`--workers 3` or `--workers 4`, and keep `--resume` enabled for long runs.

For Track 1 multimodal mode, extract the released images archive so that the
relative `img_path` values (e.g. `images/<sha>.jpg`) resolve under
`--image-root data`:

```bash
unzip -q data/images.zip       -d data/images/   # train-dev images
unzip -q data/images-testp1.zip -d data/images/  # Phase 1 test images
```

Convenience wrappers use the same defaults:

```bash
baseline_prompting/run_track1_dev.sh outputs/track1_dev_input.jsonl outputs/track1_dev_pred.jsonl
TRACK1_IMAGE_MODE=text baseline_prompting/run_track1_dev.sh
TRACK1_IMAGE_ROOT=data baseline_prompting/run_track1_dev.sh

baseline_prompting/run_track2_dev.sh outputs/track2_dev_input.jsonl outputs/track2_dev_pred.jsonl
TRACK2_MAX_INPUT_TOKENS=64000 baseline_prompting/run_track2_dev.sh
```

## Evaluate

```bash
python offline_eval/evaluate.py --track 1 \
  --gold outputs/track1_dev_gold.jsonl \
  --pred outputs/track1_dev_pred.jsonl \
  --match id

python offline_eval/evaluate.py --track 2 \
  --gold outputs/track2_dev_gold.jsonl \
  --pred outputs/track2_dev_pred.jsonl \
  --match id
```
