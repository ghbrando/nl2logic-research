from src.reasoning.claim_memory import load_ontology_memory, render_memory_prompt, retrieve_memory


def test_retrieval_uses_existing_axiom_with_provenance():
    statements = load_ontology_memory()
    retrieved = retrieve_memory("Open-source intelligence is intelligence", statements, limit=1)
    assert len(retrieved) == 1
    assert retrieved[0].kif == "(subclass OpenSourceIntelligence IntelligenceDiscipline)"
    assert retrieved[0].kind == "ontology_background"
    assert retrieved[0].source_line == 195
    assert retrieved[0].source_sha256


def test_generic_overlap_does_not_invent_context():
    statements = load_ontology_memory()
    assert retrieve_memory("Biometrics is the process", statements, limit=1) == []
    source = "translate to CNL: Biometrics is the process"
    assert render_memory_prompt(source, []) == source
