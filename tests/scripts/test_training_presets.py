import pytest

from scripts.training_presets import get_training_preset, list_training_presets


class TestTrainingPresets:
    def test_lists_expected_presets_in_order(self):
        assert [preset.name for preset in list_training_presets()] == ["smoke", "base", "full"]

    def test_base_preset_matches_current_default_run(self):
        preset = get_training_preset("base")

        assert preset.prepare_limit == 50_000
        assert preset.train_limit == 50_000
        assert preset.epochs == 3
        assert preset.batch_size == 8
        assert preset.train_file == "data/training_pairs/train_balanced_50k.jsonl"
        assert preset.output_dir == "models/flan-t5-small-cnl"

    def test_full_preset_is_uncapped(self):
        preset = get_training_preset("full")

        assert preset.prepare_limit == 0
        assert preset.train_limit == 0
        assert preset.session_name == "nl2logic-full"

    def test_unknown_preset_raises_clear_error(self):
        with pytest.raises(ValueError, match="smoke, base, full"):
            get_training_preset("overnight")