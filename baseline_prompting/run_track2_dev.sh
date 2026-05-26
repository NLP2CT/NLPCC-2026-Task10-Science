#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

INPUT="${1:-outputs/track2_dev_input.jsonl}"
OUTPUT="${2:-outputs/track2_dev_pred.jsonl}"
if (($# > 2)); then
  set -- "${@:3}"
else
  set --
fi

python3 baseline_prompting/run_baseline.py \
  --track 2 \
  --input "${INPUT}" \
  --output "${OUTPUT}" \
  --track2-max-input-tokens "${TRACK2_MAX_INPUT_TOKENS:-64000}" \
  "$@"
