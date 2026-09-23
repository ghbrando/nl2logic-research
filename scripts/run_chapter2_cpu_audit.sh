#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
state_dir="/home/save-water/nl2logic-state"
mode="${1:-raw}"
case "$mode" in
  raw)
    run_name="chapter2-cpu-inference-20260923-01"
    python_script="scripts/run_frozen_claim_predictions.py"
    ;;
  guarded)
    run_name="chapter2-cpu-guarded-20260923-01"
    python_script="scripts/run_frozen_guarded_prediction.py"
    ;;
  *)
    echo "Expected mode raw or guarded" >&2
    exit 2
    ;;
esac
run_dir="$state_dir/outputs/$run_name"

if [[ "$(id -un)" != "save-water" ]]; then
  echo "Run this audit as save-water" >&2
  exit 2
fi
if [[ -e "$run_dir" ]]; then
  echo "Output directory already exists: $run_dir" >&2
  exit 2
fi
for input in sumo_classes.jsonl sumo_relations.jsonl; do
  if [[ ! -f "$state_dir/inputs/$input" ]]; then
    echo "Missing project-owned vocabulary input: $input" >&2
    exit 2
  fi
done

export LOCAL_UID="$(id -u)" LOCAL_GID="$(id -g)"
export NL2LOGIC_STATE="$state_dir"
export XDG_RUNTIME_DIR="/run/user/$(id -u)"
systemctl --user start docker
for attempt in 1 2 3 4 5 6 7 8 9 10; do
  if docker --context rootless info >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
docker --context rootless info >/dev/null
cd "$repo_dir"
docker --context rootless compose \
  -f containers/compose.yaml -f containers/rootless.yaml \
  -f containers/inputs.yaml -f containers/cpu-audit.yaml \
  config --quiet
docker --context rootless compose \
  -f containers/compose.yaml -f containers/rootless.yaml \
  -f containers/inputs.yaml -f containers/cpu-audit.yaml \
  run --rm --no-deps \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e OMP_NUM_THREADS=2 \
  research python "$python_script" \
    --selected /workspace/data/benchmarks/fm2-0/chapter2-2023-claim-eval-source/selected_passages.jsonl \
    --selection-manifest /workspace/data/benchmarks/fm2-0/chapter2-2023-claim-eval-source/manifest.json \
    --model-path /outputs/diagnostic-closed-7k-20260922-01 \
    --output-dir "/outputs/$run_name" \
    --max-new-tokens 64
