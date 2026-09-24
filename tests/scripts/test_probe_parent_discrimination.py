from scripts.probe_parent_discrimination import build_items


def test_minimal_pairs_differ_only_in_the_gate_isolated_genus():
    items = build_items(["fusion analysis"])
    assert [(item["stated_parent"], item["symbol"]) for item in items] == [
        ("Process", "FusionAnalysis"), ("IntelligenceProduct", "FusionAnalysis")]
    assert items[0]["sentence"].startswith("Fusion analysis is the process of")
    assert items[1]["sentence"].startswith("Fusion analysis is intelligence that is produced")
