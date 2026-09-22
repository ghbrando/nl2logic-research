"""SUMO relation signatures must retain class-versus-instance constraints."""

import json

from src.sumo.loader import extract_from_file, write_relations


def test_relation_signature_kinds_survive_json_export(tmp_path):
    kif = tmp_path / "relations.kif"
    kif.write_text(
        "\n".join(
            [
                "(domain capability 2 CaseRole)",
                "(domain capability 3 Object)",
                "(domainSubclass capability 1 Process)",
                "(domainSubclass partTypes 1 Object)",
                "(domainSubclass partTypes 2 Object)",
                "(range result Entity)",
                "(rangeSubclass classResult Object)",
                # A direct domain wins over a subclass constraint on the same slot.
                "(domain mixed 1 Entity)",
                "(domainSubclass mixed 1 Object)",
            ]
        ),
        encoding="utf-8",
    )
    relations = {}
    extract_from_file(str(kif), {}, relations, {}, {})

    output = tmp_path / "relations.jsonl"
    write_relations(relations, {}, str(output))
    rows = {row["term"]: row for row in map(json.loads, output.read_text(encoding="utf-8").splitlines())}

    assert rows["capability"]["signature"] == {"1": "Process", "2": "CaseRole", "3": "Object"}
    assert rows["capability"]["signature_kinds"] == {
        "1": "subclass", "2": "instance", "3": "instance"
    }
    assert rows["partTypes"]["signature_kinds"] == {"1": "subclass", "2": "subclass"}
    assert rows["result"]["signature_kinds"] == {"range": "instance"}
    assert rows["classResult"]["signature_kinds"] == {"range": "subclass"}
    assert rows["mixed"]["signature"] == {"1": "Entity"}
    assert rows["mixed"]["signature_kinds"] == {"1": "instance"}
