from __future__ import annotations

import argparse
import subprocess
import sys

if __package__ in {None, ""}:
    from training_presets import get_training_preset, list_training_presets
else:
    from .training_presets import get_training_preset, list_training_presets

DEFAULT_HOST = "dgx-spark"
DEFAULT_REPO_DIR = "$HOME/projects/nl2logic-research"
_LOCAL_HOSTS = {"local", "localhost", "127.0.0.1", "::1"}


def _escape_for_bash_double_quotes(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _is_local_host(host: str) -> bool:
    return host.strip().lower() in _LOCAL_HOSTS


def build_dependency_check_script() -> str:
    lines = [
        "python - <<'PY'",
        "import importlib",
        "import sys",
        "",
        "missing = []",
        "for module_name in (\"torch\", \"transformers\", \"peft\", \"accelerate\", \"sentencepiece\"):",
        "    try:",
        "        importlib.import_module(module_name)",
        "    except ModuleNotFoundError:",
        "        missing.append(module_name)",
        "",
        "if missing:",
        "    sys.stderr.write(",
        "        \"Missing required training dependencies in the nl2logic environment: \"",
        "        + \", \".join(missing)",
        "        + \"\\nInstall the correct CUDA-enabled PyTorch build on the DGX before launching training.\\n\"",
        "    )",
        "    raise SystemExit(1)",
        "",
        "torch = importlib.import_module(\"torch\")",
        "if not torch.cuda.is_available():",
        "    sys.stderr.write(",
        "        \"CUDA is not available in the nl2logic environment.\\n\"",
        "        \"Install the correct CUDA-enabled PyTorch build on the DGX before launching training.\\n\"",
        "    )",
        "    raise SystemExit(1)",
        "",
        "print(f\"CUDA devices visible to PyTorch: {torch.cuda.device_count()}\")",
        "PY",
    ]
    return "\n".join(lines)


def build_optional_constrained_install_script() -> str:
    lines = [
        "python - <<'PY'",
        "import subprocess",
        "import sys",
        "",
        "subprocess.run(",
        "    [",
        "        sys.executable,",
        "        \"-m\"",
        "        , \"pip\"",
        "        , \"install\"",
        "        , \"-q\"",
        "        , \"--break-system-packages\"",
        "        , \"xgrammar\"",
        "    ],",
        "    check=True,",
        ")",
        "PY",
    ]
    return "\n".join(lines)


def build_tmux_command_body() -> str:
    parts = [
        'cd "$REPO_DIR"',
        'source "$(conda info --base)/etc/profile.d/conda.sh"',
        'conda activate nl2logic',
        'mkdir -p "$LOG_DIR"',
        'python src/training/train.py --train-file "$TRAIN_FILE" --output-dir "$OUTPUT_DIR" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --learning-rate "$LEARNING_RATE" --limit "$TRAIN_LIMIT" > "$LOG_FILE" 2>&1',
    ]
    return " && ".join(parts)


def build_remote_script(
    preset_name: str,
    repo_dir: str = DEFAULT_REPO_DIR,
    *,
    allow_gold_cnl_overlap: bool = False,
) -> str:
    preset = get_training_preset(preset_name)
    repo_dir_value = _escape_for_bash_double_quotes(repo_dir)
    continuation = "\\"
    prepare_args = "--allow-gold-cnl-overlap" if allow_gold_cnl_overlap else ""
    leakage_args = "--ignore-cnl-overlap" if allow_gold_cnl_overlap else ""
    lines = [
        "set -euo pipefail",
        "",
        f'REPO_DIR="{repo_dir_value}"',
        f'PRESET_NAME="{preset.name}"',
        f'SESSION_NAME="{preset.session_name}"',
        f'TRAIN_FILE="{preset.train_file}"',
        f'OUTPUT_DIR="{preset.output_dir}"',
        'LOG_DIR="logs"',
        f'PREPARE_LIMIT="{preset.prepare_limit}"',
        f'TRAIN_LIMIT="{preset.train_limit}"',
        f'EPOCHS="{preset.epochs}"',
        f'BATCH_SIZE="{preset.batch_size}"',
        f'LEARNING_RATE="{preset.learning_rate}"',
        f'PREPARE_ARGS="{prepare_args}"',
        f'LEAKAGE_ARGS="{leakage_args}"',
        "",
        'cd "$REPO_DIR"',
        'source "$(conda info --base)/etc/profile.d/conda.sh"',
        'conda activate nl2logic',
        "",
        'pip install -q --break-system-packages -r requirements.txt',
        'pip install -q --break-system-packages transformers peft accelerate sentencepiece',
        build_optional_constrained_install_script(),
        "",
        build_dependency_check_script(),
        "",
        'if [ ! -f "$TRAIN_FILE" ]; then',
        '    python scripts/prepare_training_data.py --limit "$PREPARE_LIMIT" --output "$TRAIN_FILE" $PREPARE_ARGS',
        'fi',
        "",
        'if ! python scripts/check_leakage.py --train-path "$TRAIN_FILE" $LEAKAGE_ARGS; then',
        '    echo "Leakage check failed. Aborting before training." >&2',
        '    exit 1',
        'fi',
        "",
        'mkdir -p "$LOG_DIR"',
        'tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true',
        "",
        'TIMESTAMP="$(date +%Y%m%d_%H%M%S)"',
        'LOG_FILE="$LOG_DIR/training_${PRESET_NAME}_${TIMESTAMP}.log"',
        'RUN_INFO_FILE="$LOG_DIR/training_${PRESET_NAME}_${TIMESTAMP}.meta"',
        "",
        f"printf '%s\\n' {continuation}",
        f'    "preset=$PRESET_NAME" {continuation}',
        f'    "train_file=$TRAIN_FILE" {continuation}',
        f'    "output_dir=$OUTPUT_DIR" {continuation}',
        f'    "epochs=$EPOCHS" {continuation}',
        f'    "batch_size=$BATCH_SIZE" {continuation}',
        f'    "learning_rate=$LEARNING_RATE" {continuation}',
        f'    "prepare_limit=$PREPARE_LIMIT" {continuation}',
        '    "train_limit=$TRAIN_LIMIT" > "$RUN_INFO_FILE"',
        "",
        'TMUX_COMMAND="$(cat <<EOF',
        build_tmux_command_body(),
        'EOF',
        ')"',
        'tmux new-session -d -s "$SESSION_NAME" "$TMUX_COMMAND"',
        "",
        'echo "Preset: $PRESET_NAME"',
        'echo "Remote output dir: $REPO_DIR/$OUTPUT_DIR"',
        'echo "Remote log file: $REPO_DIR/$LOG_FILE"',
        'echo "Run metadata: $REPO_DIR/$RUN_INFO_FILE"',
    ]
    return "\n".join(lines)


def build_attach_command(host: str, preset_name: str) -> str:
    preset = get_training_preset(preset_name)
    if _is_local_host(host):
        return f"tmux attach -t {preset.session_name}"
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
        help=f"SSH host for remote training, or 'local' to run on the current machine (default: {DEFAULT_HOST})",
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
    parser.add_argument(
        "--allow-gold-cnl-overlap",
        action="store_true",
        help="Keep paraphrases whose CNL matches the gold eval set while still excluding exact gold NL and probe NL overlaps.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def print_presets() -> None:
    print("Available training presets:")
    for preset in list_training_presets():
        print(f"  {preset.name:5s} {preset.description}")


def run_remote_script(host: str, remote_script: str) -> None:
    command = ["bash", "-s"] if _is_local_host(host) else ["ssh", host, "bash -s"]
    try:
        subprocess.run(
            command,
            input=remote_script,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        if not _is_local_host(host) and exc.returncode == 255:
            raise RuntimeError(
                f"Failed to SSH to {host!r}. If you are already on the target machine, rerun with '--host local'."
            ) from exc
        raise


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list_presets:
        print_presets()
        return 0

    remote_script = build_remote_script(
        args.preset,
        args.repo_dir,
        allow_gold_cnl_overlap=args.allow_gold_cnl_overlap,
    )
    try:
        run_remote_script(args.host, remote_script)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"Training started. Attach with: {build_attach_command(args.host, args.preset)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())





