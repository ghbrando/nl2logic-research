from __future__ import annotations

import argparse
import subprocess

if __package__ in {None, ""}:
    from training_presets import get_training_preset, list_training_presets
else:
    from .training_presets import get_training_preset, list_training_presets

DEFAULT_HOST = "dgx-spark"
DEFAULT_REPO_DIR = "$HOME/projects/nl2logic-research"


def _escape_for_bash_double_quotes(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def build_dependency_check_script() -> str:
    return '''python - <<'PY'
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
PY'''


def build_remote_script(preset_name: str, repo_dir: str = DEFAULT_REPO_DIR) -> str:
    preset = get_training_preset(preset_name)
    repo_dir_value = _escape_for_bash_double_quotes(repo_dir)
    return f'''set -euo pipefail

REPO_DIR="{repo_dir_value}"
PRESET_NAME="{preset.name}"
SESSION_NAME="{preset.session_name}"
TRAIN_FILE="{preset.train_file}"
OUTPUT_DIR="{preset.output_dir}"
LOG_DIR="logs"
PREPARE_LIMIT="{preset.prepare_limit}"
TRAIN_LIMIT="{preset.train_limit}"
EPOCHS="{preset.epochs}"
BATCH_SIZE="{preset.batch_size}"
LEARNING_RATE="{preset.learning_rate}"

cd "$REPO_DIR"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nl2logic

pip install -q --break-system-packages -r requirements.txt
pip install -q --break-system-packages transformers peft accelerate sentencepiece

{build_dependency_check_script()}

if [ ! -f "$TRAIN_FILE" ]; then
    python scripts/prepare_training_data.py --limit "$PREPARE_LIMIT" --output "$TRAIN_FILE"
fi

if ! python scripts/check_leakage.py --train-path "$TRAIN_FILE"; then
    echo "Leakage check failed. Aborting before training." >&2
    exit 1
fi

mkdir -p "$LOG_DIR"
tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/training_${{PRESET_NAME}}_${{TIMESTAMP}}.log"
RUN_INFO_FILE="$LOG_DIR/training_${{PRESET_NAME}}_${{TIMESTAMP}}.meta"

printf '%s\n' \
    "preset=$PRESET_NAME" \
    "train_file=$TRAIN_FILE" \
    "output_dir=$OUTPUT_DIR" \
    "epochs=$EPOCHS" \
    "batch_size=$BATCH_SIZE" \
    "learning_rate=$LEARNING_RATE" \
    "prepare_limit=$PREPARE_LIMIT" \
    "train_limit=$TRAIN_LIMIT" > "$RUN_INFO_FILE"

tmux new-session -d -s "$SESSION_NAME" \
    "cd \"$REPO_DIR\" && source \"$(conda info --base)/etc/profile.d/conda.sh\" && conda activate nl2logic && mkdir -p \"$LOG_DIR\" && python src/training/train.py --train-file \"$TRAIN_FILE\" --output-dir \"$OUTPUT_DIR\" --epochs \"$EPOCHS\" --batch-size \"$BATCH_SIZE\" --learning-rate \"$LEARNING_RATE\" --limit \"$TRAIN_LIMIT\" > \"$LOG_FILE\" 2>&1"

echo "Preset: $PRESET_NAME"
echo "Remote output dir: $REPO_DIR/$OUTPUT_DIR"
echo "Remote log file: $REPO_DIR/$LOG_FILE"
echo "Run metadata: $REPO_DIR/$RUN_INFO_FILE"'''.strip()


def build_attach_command(host: str, preset_name: str) -> str:
    preset = get_training_preset(preset_name)
    return f"ssh {host} -t tmux attach -t {preset.session_name}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch a preset NL2Logic training run on the DGX.")
    parser.add_argument(
        "preset",
        nargs="?",
        default="base",
        choices=[preset.name for preset in list_training_presets()],
        help="Training preset to launch (default: base)",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"SSH host for remote training (default: {DEFAULT_HOST})",
    )
    parser.add_argument(
        "--repo-dir",
        default=DEFAULT_REPO_DIR,
        help=f"Remote repository path (default: {DEFAULT_REPO_DIR})",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="Print available presets and exit.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def print_presets() -> None:
    print("Available training presets:")
    for preset in list_training_presets():
        print(f"  {preset.name:5s} {preset.description}")


def run_remote_script(host: str, remote_script: str) -> None:
    subprocess.run(
        ["ssh", host, "bash -s"],
        input=remote_script,
        text=True,
        check=True,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list_presets:
        print_presets()
        return 0

    remote_script = build_remote_script(args.preset, args.repo_dir)
    run_remote_script(args.host, remote_script)
    print(f"Training started. Attach with: {build_attach_command(args.host, args.preset)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())