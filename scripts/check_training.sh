#!/usr/bin/env bash
set -euo pipefail

HOST="dgx-spark"

ssh "$HOST" "bash -s" <<'REMOTE'
set -e

REPO_DIR="$HOME/projects/nl2logic-research"
SESSION_NAME="nl2logic-full"

cd "$REPO_DIR"

LATEST_LOG="$(ls -1t logs/training_*.log 2>/dev/null | head -n 1 || true)"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Training session is running: $SESSION_NAME"
else
    echo "Training session not found"
fi

if [ -n "$LATEST_LOG" ]; then
    tail -n 40 "$LATEST_LOG"
else
    echo "No training log files found."
fi
REMOTE
