#!/usr/bin/env bash
set -euo pipefail

python offline_eval/evaluate.py --track 1 --gold "$1" --pred "$2" "${@:3}"

