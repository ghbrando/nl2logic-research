from pathlib import Path

import pytest

import scripts.check_training as check_training
from scripts.check_training import build_remote_script, parse_args


class TestCheckTrainingWrapper:
    def test_shell_wrapper_delegates_to_python_entrypoint(self):
        script = Path("scripts/check_training.sh").read_text(encoding="utf-8")

        assert 'python scripts/check_training.py "$@"' in script


class TestCheckTrainingRemoteScript:
    def test_base_preset_tracks_preset_specific_session_and_logs(self):
        script = build_remote_script("base")

        assert 'SESSION_NAME="nl2logic-base"' in script
        assert 'training_${PRESET_NAME}_*.log' in script
        assert 'Training session is running: $SESSION_NAME' in script
        assert 'Latest log: $LATEST_LOG' in script

    def test_parse_args_defaults_to_base_preset(self):
        args = parse_args([])

        assert args.preset == "base"
        assert args.host == "dgx-spark"


class TestCheckTrainingExecution:
    def test_run_remote_script_uses_local_bash_for_local_host(self, monkeypatch):
        seen = {}

        def fake_run(command, *, input, text, check):
            seen["command"] = command
            seen["input"] = input
            seen["text"] = text
            seen["check"] = check

        monkeypatch.setattr(check_training.subprocess, "run", fake_run)

        check_training.run_remote_script("local", "echo ok")

        assert seen["command"] == ["bash", "-s"]
        assert seen["input"] == "echo ok"
        assert seen["text"] is True
        assert seen["check"] is True

    def test_run_remote_script_raises_helpful_error_for_failed_ssh(self, monkeypatch):
        def fake_run(*_args, **_kwargs):
            raise check_training.subprocess.CalledProcessError(255, ["ssh", "dgx-spark", "bash -s"])

        monkeypatch.setattr(check_training.subprocess, "run", fake_run)

        with pytest.raises(RuntimeError, match="--host local"):
            check_training.run_remote_script("dgx-spark", "echo ok")