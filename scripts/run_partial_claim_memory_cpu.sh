#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
state_dir="/home/save-water/nl2logic-state"
prefix="partial-claim-memory-cpu-20260923"
run_mode="${1:-paired}"

if [[ "$(id -un)" != "save-water" ]]; then
  echo "Run this memory probe as save-water" >&2
  exit 2
fi
if [[ "$run_mode" == "paired" ]]; then
  for mode in baseline retrieved; do
    if [[ -e "$state_dir/outputs/$prefix-$mode" ]]; then
      echo "Output directory already exists: $state_dir/outputs/$prefix-$mode" >&2
      exit 2
    fi
  done
elif [[ "$run_mode" == "control" ]]; then
  if [[ -e "$state_dir/outputs/$prefix-control" ]]; then
    echo "Output directory already exists: $state_dir/outputs/$prefix-control" >&2
    exit 2
  fi
elif [[ "$run_mode" == "declarations" || "$run_mode" == "definition-gate" ]]; then
  if [[ -e "$state_dir/outputs/$prefix-$run_mode" ]]; then
    echo "Output directory already exists: $state_dir/outputs/$prefix-$run_mode" >&2
    exit 2
  fi
else
  echo "Expected paired, control, declarations, or definition-gate" >&2
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
if [[ "$run_mode" == "control" ]]; then
  docker --context rootless compose \
    -f containers/compose.yaml -f containers/rootless.yaml \
    -f containers/inputs.yaml -f containers/cpu-audit.yaml \
    run --rm --no-deps \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e OMP_NUM_THREADS=2 \
    research python scripts/run_claim_memory_control.py \
      --model-path /outputs/diagnostic-closed-7k-20260922-01 \
      --output-dir "/outputs/$prefix-control"
  exit 0
fi
if [[ "$run_mode" == "declarations" || "$run_mode" == "definition-gate" ]]; then
  docker --context rootless compose \
    -f containers/compose.yaml -f containers/rootless.yaml \
    -f containers/inputs.yaml -f containers/cpu-audit.yaml \
    run --rm --no-deps \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e OMP_NUM_THREADS=2 \
    research python scripts/run_partial_claim_declaration_probe.py \
      --sources /workspace/data/benchmarks/fm2-0/chapter1-partial-claim-development-source/sources.jsonl \
      --source-manifest /workspace/data/benchmarks/fm2-0/chapter1-partial-claim-development-source/manifest.json \
      --registry /workspace/data/benchmarks/fm2-0/chapter1-partial-claim-development-source/provisional_declarations.json \
      --model-path /outputs/diagnostic-closed-7k-20260922-01 \
      --output-dir "/outputs/$prefix-$run_mode" \
      --max-new-tokens 64
  exit 0
fi
for mode in baseline retrieved; do
  memory_mode="none"
  if [[ "$mode" == "retrieved" ]]; then
    memory_mode="retrieved"
  fi
  docker --context rootless compose \
    -f containers/compose.yaml -f containers/rootless.yaml \
    -f containers/inputs.yaml -f containers/cpu-audit.yaml \
    run --rm --no-deps \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e OMP_NUM_THREADS=2 \
    research python scripts/run_partial_claim_dev_probe.py \
      --sources /workspace/data/benchmarks/fm2-0/chapter1-partial-claim-development-source/sources.jsonl \
      --source-manifest /workspace/data/benchmarks/fm2-0/chapter1-partial-claim-development-source/manifest.json \
      --model-path /outputs/diagnostic-closed-7k-20260922-01 \
      --output-dir "/outputs/$prefix-$mode" \
      --memory-mode "$memory_mode" \
      --max-new-tokens 64
done
