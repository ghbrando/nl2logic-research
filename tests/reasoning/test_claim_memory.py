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


def test_prior_claims_are_retrieved_by_genus_head_only():
    from src.reasoning.claim_memory import MemoryStatement, retrieve_prior_by_genus

    prior = [MemoryStatement("p:1", "(subclass BiometricsProcess Process)", "BiometricsProcess", "Process",
                             "prior.jsonl", 1, "0" * 64, kind="prior_accepted_claim"),
             MemoryStatement("p:2", "(subclass OsintProduct IntelligenceProduct)", "OsintProduct",
                             "IntelligenceProduct", "prior.jsonl", 2, "0" * 64, kind="prior_accepted_claim")]
    assert [s.statement_id for s in retrieve_prior_by_genus("army s primary process", prior)] == ["p:1"]
    assert [s.statement_id for s in retrieve_prior_by_genus("intelligence product", prior)] == ["p:2"]
    assert retrieve_prior_by_genus("display", prior) == []
