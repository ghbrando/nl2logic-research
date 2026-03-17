from pathlib import Path

from scripts.launch_training import build_attach_command, build_remote_script, parse_args


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
        assert '--limit' in script

    def test_keeps_dependency_and_cuda_checks(self):
        script = build_remote_script("base")

        assert 'transformers' in script
        assert 'torch.cuda.is_available()' in script
        assert 'CUDA is not available in the nl2logic environment.' in script

    def test_attach_command_uses_preset_session_name(self):
        assert build_attach_command("dgx-spark", "base") == "ssh dgx-spark -t tmux attach -t nl2logic-base"

    def test_parse_args_defaults_to_base_preset(self):
        args = parse_args([])

        assert args.preset == "base"
        assert args.host == "dgx-spark"
        assert args.list_presets is False