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


def build_remote_script(preset_name: str, repo_dir: str = DEFAULT_REPO_DIR) -> str:
    preset = get_training_preset(preset_name)
    repo_dir_value = _escape_for_bash_double_quotes(repo_dir)
    return f'''set -euo pipefail

REPO_DIR="{repo_dir_value}"
PRESET_NAME="{preset.name}"
SESSION_NAME="{preset.session_name}"
LOG_DIR="logs"

cd "$REPO_DIR"

LATEST_LOG="$(ls -1t "$LOG_DIR"/training_${{PRESET_NAME}}_*.log 2>/dev/null | head -n 1 || true)"
LATEST_META=""
if [ -n "$LATEST_LOG" ]; then
    LATEST_META="${{LATEST_LOG%.log}}.meta"
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Training session is running: $SESSION_NAME"
else
    echo "Training session not found: $SESSION_NAME"
fi

if [ -n "$LATEST_META" ] && [ -f "$LATEST_META" ]; then
    echo "Run metadata:"
    cat "$LATEST_META"
fi

if [ -n "$LATEST_LOG" ]; then
    echo "Latest log: $LATEST_LOG"
    tail -n 40 "$LATEST_LOG"
else
    echo "No training log files found for preset: $PRESET_NAME"
fi'''.strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check a preset NL2Logic training run on the DGX.")
    parser.add_argument(
        "preset",
        nargs="?",
        default="base",
        choices=[preset.name for preset in list_training_presets()],
        help="Training preset to inspect (default: base)",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"SSH host for remote training, or 'local' to inspect the current machine (default: {DEFAULT_HOST})",
    )
    parser.add_argument(
        "--repo-dir",
        default=DEFAULT_REPO_DIR,
        help=f"Remote repository path (default: {DEFAULT_REPO_DIR})",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


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
    try:
        run_remote_script(args.host, build_remote_script(args.preset, args.repo_dir))
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())