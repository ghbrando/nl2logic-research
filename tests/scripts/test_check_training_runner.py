from pathlib import Path

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