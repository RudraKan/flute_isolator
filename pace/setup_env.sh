#!/usr/bin/env bash
# One-time setup on a PACE login node (needs internet). Creates .venv in the bundle folder.
# Run from the unzipped folder:  bash pace/setup_env.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# keep the multi-GB torch download out of the small home quota
if [ -d "$HOME/scratch" ]; then
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/scratch/.cache/uv}"
  export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$HOME/scratch/.cache/pip}"
fi

if command -v uv >/dev/null 2>&1 || [ -x "$HOME/.local/bin/uv" ] || curl -LsSf https://astral.sh/uv/install.sh | sh; then
  export PATH="$HOME/.local/bin:$PATH"
  uv venv --python 3.12 .venv
  uv pip install --python .venv/bin/python -r pace/requirements.txt
else
  # fallback without uv: the cluster's python module
  module load python/3.12 2>/dev/null || module load anaconda3 2>/dev/null || true
  python3 -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install -r pace/requirements.txt
fi

.venv/bin/python - <<'PY'
import torch
print("torch", torch.__version__, "| CUDA build", torch.version.cuda, "| GPU visible here:", torch.cuda.is_available())
PY
echo "Environment ready. Submit with:  sbatch pace/train.sbatch"
