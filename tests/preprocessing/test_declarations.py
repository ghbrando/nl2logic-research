import json
from pathlib import Path

import pytest

from src.preprocessing.declarations import load_declaration_registry, match_declaration


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/benchmarks/fm2-0/chapter1-partial-claim-development-source"


def _registry():
    manifest = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    return load_declaration_registry(
        SOURCE / "provisional_declarations.json",
        source_packet_sha256=manifest["packet_sha256"],
    )


def test_registry_contains_only_isolated_new_class_declarations():
    registry = _registry()
    assert len(registry["declarations"]) == 3
    assert all("kif" not in entry and "cnl" not in entry for entry in registry["declarations"])
    assert registry["existing_parent_pool"] == ["Process", "IntelligenceProduct"]


def test_declaration_requires_matching_source_evidence():
    registry = _registry()
    assert match_declaration(
        "Biometrics is the process",
        "Biometrics is the process of recognizing an individual.",
        registry,
    )["symbol"] == "BiometricsProcess"
    assert match_declaration("Biometrics is the process", "Biometrics is a characteristic.", registry) is None


def test_registry_rejects_wrong_source_packet():
    with pytest.raises(ValueError, match="not tied"):
        load_declaration_registry(SOURCE / "provisional_declarations.json", source_packet_sha256="0" * 64)
