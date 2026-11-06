#!/usr/bin/env bash
# Same training without SLURM, for an interactive GPU session (Open OnDemand Jupyter/desktop or `salloc --gres=gpu:1`).
set -euo pipefail
cd "$(dirname "$0")/.."
RUN="${RUN:-notenet-$(date +%Y%m%d-%H%M)}"
NO_OVERRIDES='{}'
mkdir -p runs
.venv/bin/python -m sargam.ml.train --corpus data/corpus/songs.jsonl --eval-dir data/eval --out "runs/$RUN" \
  --steps "${STEPS:-60000}" --batch "${BATCH:-64}" --workers "${WORKERS:-8}" --eval-every 2000 --model "${MODEL:-$NO_OVERRIDES}" "$@"
