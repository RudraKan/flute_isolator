#!/bin/bash
# Train NoteNet on one GPU. Submit from the unzipped bundle folder:  sbatch pace/train.sbatch
# Override settings with environment variables, e.g.:
#   STEPS=100000 BATCH=96 MODEL='{"layers": 6, "d_model": 320, "heads": 5}' sbatch pace/train.sbatch
#SBATCH -J notenet
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -t 8:00:00
#SBATCH -o notenet-%j.out
## Pick a GPU type if you want one, e.g.  --gres=gpu:H100:1  or  --gres=gpu:A100:1  or  --gres=gpu:V100:1
## Phoenix (research cluster) needs a charge account and QoS; ICE (instructional) does not:
##SBATCH -A gts-<your-pi-account>
##SBATCH -q inferno
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
RUN="${RUN:-notenet-$(date +%Y%m%d-%H%M)-${SLURM_JOB_ID:-local}}"
STEPS="${STEPS:-60000}"
BATCH="${BATCH:-64}"
NO_OVERRIDES='{}'
MODEL="${MODEL:-$NO_OVERRIDES}"
EXTRA="${EXTRA:-}"
nvidia-smi || true
mkdir -p runs
.venv/bin/python -m sargam.ml.train \
  --corpus data/corpus/songs.jsonl --eval-dir data/eval --out "runs/$RUN" \
  --steps "$STEPS" --batch "$BATCH" --workers "${SLURM_CPUS_PER_TASK:-8}" --eval-every 2000 \
  --model "$MODEL" $EXTRA
echo "checkpoint: runs/$RUN/best.pt"
