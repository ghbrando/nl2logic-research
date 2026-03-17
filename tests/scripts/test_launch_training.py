from pathlib import Path


LAUNCH_SCRIPT = Path("scripts/launch_training.sh")


class TestLaunchTrainingScript:
    def test_does_not_require_optional_ingest_dependencies(self):
        script = LAUNCH_SCRIPT.read_text(encoding="utf-8")

        assert "python -m spacy download en_core_web_lg" not in script
        assert "python -m coreferee install en" not in script

    def test_refuses_to_launch_without_cuda(self):
        script = LAUNCH_SCRIPT.read_text(encoding="utf-8")

        assert "torch.cuda.is_available()" in script
        assert "CUDA is not available in the nl2logic environment." in script
