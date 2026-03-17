from pathlib import Path
import ast

import pytest

import scripts.launch_training as launch_training
from scripts.launch_training import (
    build_attach_command,
    build_dependency_check_script,
    build_optional_constrained_install_script,
    build_remote_script,
    parse_args,
)


class TestLaunchTrainingWrapper:
    def test_shell_wrapper_delegates_to_python_entrypoint(self):
        script = Path("scripts/launch_training.sh").read_text(encoding="utf-8")

        assert 'python scripts/launch_training.py "$@"' in script


class TestLaunchTrainingRemoteScript:
    def test_smoke_preset_uses_small_artifact_and_session(self):
        script = build_remote_script("smoke")

        assert 'SESSION_NAME="nl2logic-smoke"' in script
        assert 'TRAIN_FILE="data/training_pairs/train_balanced_7k.jsonl"' in script
        assert 'OUTPUT_DIR="models/flan-t5-small-cnl-smoke"' in script
        assert 'PREPARE_LIMIT="7000"' in script
        assert 'TRAIN_LIMIT="7000"' in script
        assert 'python scripts/check_leakage.py --train-path "$TRAIN_FILE"' in script

    def test_full_preset_uses_uncapped_training_and_full_output_dir(self):
        script = build_remote_script("full")

        assert 'SESSION_NAME="nl2logic-full"' in script
        assert 'TRAIN_FILE="data/training_pairs/train_full.jsonl"' in script
        assert 'OUTPUT_DIR="models/flan-t5-small-cnl-full"' in script
        assert 'PREPARE_LIMIT="0"' in script
        assert 'TRAIN_LIMIT="0"' in script
        assert '--limit "$TRAIN_LIMIT"' in script

    def test_keeps_dependency_and_cuda_checks(self):
        script = build_remote_script("base")

        assert 'transformers' in script
        assert 'torch.cuda.is_available()' in script
        assert 'CUDA is not available in the nl2logic environment.' in script

    def test_dependency_check_embeds_valid_python(self):
        script = build_dependency_check_script()
        python_body = script.split("\n", 1)[1].rsplit("\nPY", 1)[0]

        ast.parse(python_body)

    def test_optional_constrained_install_script_embeds_valid_python(self):
        script = build_optional_constrained_install_script()
        python_body = script.split("\n", 1)[1].rsplit("\nPY", 1)[0]

        ast.parse(python_body)
        assert 'xgrammar' in script
        assert 'outlines>=1.0' not in script

    def test_remote_script_uses_heredoc_tmux_command(self):
        script = build_remote_script("base")

        assert 'TMUX_COMMAND="$(cat <<EOF' in script
        assert 'tmux new-session -d -s "$SESSION_NAME" "$TMUX_COMMAND"' in script

    def test_remote_script_can_allow_gold_cnl_overlap(self):
        script = build_remote_script("base", allow_gold_cnl_overlap=True)

        assert '--allow-gold-cnl-overlap' in script
        assert '--ignore-cnl-overlap' in script

    def test_attach_command_uses_preset_session_name(self):
        assert build_attach_command("dgx-spark", "base") == "ssh dgx-spark -t tmux attach -t nl2logic-base"

    def test_attach_command_uses_plain_tmux_for_local_runs(self):
        assert build_attach_command("local", "smoke") == "tmux attach -t nl2logic-smoke"

    def test_parse_args_defaults_to_base_preset(self):
        args = parse_args([])

        assert args.preset == "base"
        assert args.host == "dgx-spark"
        assert args.list_presets is False


class TestLaunchTrainingExecution:
    def test_run_remote_script_uses_local_bash_for_local_host(self, monkeypatch):
        seen = {}

        def fake_run(command, *, input, text, check):
            seen["command"] = command
            seen["input"] = input
            seen["text"] = text
            seen["check"] = check

        monkeypatch.setattr(launch_training.subprocess, "run", fake_run)

        launch_training.run_remote_script("local", "echo ok")

        assert seen["command"] == ["bash", "-s"]
        assert seen["input"] == "echo ok"
        assert seen["text"] is True
        assert seen["check"] is True

    def test_run_remote_script_raises_helpful_error_for_failed_ssh(self, monkeypatch):
        def fake_run(*_args, **_kwargs):
            raise launch_training.subprocess.CalledProcessError(255, ["ssh", "dgx-spark", "bash -s"])

        monkeypatch.setattr(launch_training.subprocess, "run", fake_run)

        with pytest.raises(RuntimeError, match="--host local"):
            launch_training.run_remote_script("dgx-spark", "echo ok")



