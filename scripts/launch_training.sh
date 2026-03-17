#!/usr/bin/env bash
set -euo pipefail

HOST="dgx-spark"

ssh "$HOST" "bash -s" <<'REMOTE'
set -e

# ---------------------------------------------------------------------------
# Remote configuration
# ---------------------------------------------------------------------------
REPO_DIR="$HOME/projects/nl2logic-research"
SESSION_NAME="nl2logic-full"
TRAIN_FILE="data/training_pairs/train_balanced_50k.jsonl"
LOG_DIR="logs"

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
cd "$REPO_DIR"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nl2logic

pip install -q --break-system-packages -r requirements.txt
pip install -q --break-system-packages transformers peft accelerate sentencepiece

python - <<'PY'
import importlib
import sys

missing = []
for module_name in ("torch", "transformers", "peft", "accelerate", "sentencepiece"):
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError:
        missing.append(module_name)

if missing:
    sys.stderr.write(
        "Missing required training dependencies in the nl2logic environment: "
        + ", ".join(missing)
        + "\nInstall the correct CUDA-enabled PyTorch build on the DGX before launching training.\n"
    )
    raise SystemExit(1)

torch = importlib.import_module("torch")
if not torch.cuda.is_available():
    sys.stderr.write(
        "CUDA is not available in the nl2logic environment.\n"
        "Install the correct CUDA-enabled PyTorch build on the DGX before launching training.\n"
    )
    raise SystemExit(1)

print(f"CUDA devices visible to PyTorch: {torch.cuda.device_count()}")
PY

# ---------------------------------------------------------------------------
# Prepared artifact + leakage check
# ---------------------------------------------------------------------------
if [ ! -f "$TRAIN_FILE" ]; then
    python scripts/prepare_training_data.py --limit 50000
fi

if ! python scripts/check_leakage.py; then
    echo "Leakage check failed. Aborting before training." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Launch detached tmux training session
# ---------------------------------------------------------------------------
mkdir -p "$LOG_DIR"
tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/training_${TIMESTAMP}.log"

tmux new-session -d -s "$SESSION_NAME" \
    "cd \"$REPO_DIR\" && source \"$(conda info --base)/etc/profile.d/conda.sh\" && conda activate nl2logic && mkdir -p \"$LOG_DIR\" && python src/training/train.py --train-file \"$TRAIN_FILE\" --epochs 3 > \"$LOG_FILE\" 2>&1"

echo "Remote log file: $REPO_DIR/$LOG_FILE"
REMOTE

echo "Training started. Attach with: ssh dgx-spark -t tmux attach -t nl2logic-full"

