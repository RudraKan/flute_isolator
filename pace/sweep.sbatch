#!/bin/bash
# Four model sizes in parallel (a SLURM job array, one GPU each). Compare runs/sweep-*/metrics.jsonl afterwards
# with:  .venv/bin/python pace/summarize.py runs/sweep-*
#SBATCH -J notenet-sweep
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -t 8:00:00
#SBATCH --array=0-3
#SBATCH -o notenet-sweep-%A_%a.out
##SBATCH -A gts-<your-pi-account>
##SBATCH -q inferno
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
CONFIGS=(
  '{}'
  '{"layers": 6, "d_model": 320, "heads": 5}'
  '{"layers": 8, "d_model": 384, "heads": 6, "channels": 24}'
  '{"temporal": "gru", "layers": 2, "hidden": 256}'
)
NAMES=(base medium large gru)
i="${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p runs
.venv/bin/python -m sargam.ml.train \
  --corpus data/corpus/songs.jsonl --eval-dir data/eval --out "runs/sweep-${NAMES[$i]}" \
  --steps "${STEPS:-60000}" --batch "${BATCH:-64}" --workers "${SLURM_CPUS_PER_TASK:-8}" --eval-every 2000 \
  --model "${CONFIGS[$i]}"
