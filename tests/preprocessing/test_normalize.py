from __future__ import annotations

import importlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.preprocessing.normalize import (
    NLPAdapter,
    NormalizationError,
    NormalizationResult,
    build_prompt,
    normalize,
    parse_response,
)

normalize_module = importlib.import_module("src.preprocessing.normalize")


class _FakeToken:
    def __init__(
        self,
        text: str,
        *,
        pos_: str = "",
        dep_: str = "",
        children: list[_FakeToken] | None = None,
        resolved_text: str | None = None,
    ) -> None:
        self.text = text
        self.pos_ = pos_
        self.dep_ = dep_
        self.children = children or []
        self._ = SimpleNamespace(resolved_text=resolved_text)


class _FakeSpan:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeDoc:
    def __init__(self, tokens: list[_FakeToken], *, noun_chunks: list[str] | None = None) -> None:
        self._tokens = tokens
        self.noun_chunks = [_FakeSpan(text) for text in (noun_chunks or [])]
        self._ = SimpleNamespace(coref_chains=None)

    def __iter__(self):
        return iter(self._tokens)


class _FakeNLP:
    def __init__(self, docs_by_text: dict[str, _FakeDoc]) -> None:
        self._docs_by_text = docs_by_text
        self.pipe_names: list[str] = []
        self.added_pipes: list[str] = []

    def __call__(self, text: str) -> _FakeDoc:
        return self._docs_by_text[text]

    def add_pipe(self, name: str) -> None:
        self.added_pipes.append(name)
        self.pipe_names.append(name)


class _FakeSpacyModule:
    def __init__(self, docs_by_text: dict[str, _FakeDoc]) -> None:
        self._docs_by_text = docs_by_text
        self.fake_nlp = _FakeNLP(docs_by_text)

    def load(self, model_name: str) -> _FakeNLP:
        assert model_name == "en_core_web_lg"
        return self.fake_nlp


class TestBuildPrompt:
    def test_sentence_and_context_appear_in_prompt(self):
        prompt = build_prompt("It reports.", ["The S2 observes the AO."])

        assert "It reports." in prompt
        assert "The S2 observes the AO." in prompt

    def test_empty_context_produces_valid_prompt(self):
        prompt = build_prompt("Conducts ISR.", [])

        assert "Context window:" in prompt
        assert "(none)" in prompt
        assert "Conducts ISR." in prompt


class TestParseResponse:
    def test_single_clean_line_goes_to_normalized(self):
        result = parse_response("The S2 conducts ISR.", original="Conducts ISR.")

        assert result == NormalizationResult(
            normalized=["The S2 conducts ISR."],
            ambiguous=[],
            original="Conducts ISR.",
        )

    def test_ambiguous_line_goes_to_ambiguous_without_prefix(self):
        result = parse_response("[AMBIGUOUS] Conducts ISR.", original="Conducts ISR.")

        assert result == NormalizationResult(
            normalized=[],
            ambiguous=["Conducts ISR."],
            original="Conducts ISR.",
        )

    def test_mixed_output_partitions_normalized_and_ambiguous(self):
        response = "The S2 conducts ISR.\n[AMBIGUOUS] Conducts ISR."

        result = parse_response(response, original="Conducts ISR.")

        assert result == NormalizationResult(
            normalized=["The S2 conducts ISR."],
            ambiguous=["Conducts ISR."],
            original="Conducts ISR.",
        )

    def test_empty_string_treats_original_as_ambiguous(self):
        result = parse_response("", original="Conducts ISR.")

        assert result == NormalizationResult(
            normalized=[],
            ambiguous=["Conducts ISR."],
            original="Conducts ISR.",
        )


class TestNormalize:
    def test_provider_receives_correct_sentence_and_context(self):
        provider = Mock(return_value="The S2 conducts ISR.")

        result = normalize("Conducts ISR.", ["The S2 observes the AO."], provider)

        provider.assert_called_once_with("Conducts ISR.", ["The S2 observes the AO."])
        assert result.normalized == ["The S2 conducts ISR."]
        assert result.ambiguous == []

    def test_normalization_error_returns_fail_closed_result(self):
        def provider(sentence: str, context: list[str]) -> str:
            raise NormalizationError("backend unavailable")

        result = normalize("Conducts ISR.", ["The S2 observes the AO."], provider)

        assert result == NormalizationResult(
            normalized=[],
            ambiguous=["Conducts ISR."],
            original="Conducts ISR.",
        )

    def test_clean_provider_output_returns_normalized_sentences(self):
        result = normalize(
            "Conducts ISR.",
            [],
            lambda sentence, context: "The S2 conducts ISR.\nThe S2 reports.",
        )

        assert result == NormalizationResult(
            normalized=["The S2 conducts ISR.", "The S2 reports."],
            ambiguous=[],
            original="Conducts ISR.",
        )


class TestNLPAdapter:
    def test_pronoun_in_sentence_is_resolved_from_context(self, monkeypatch):
        sentence = "It conducts ISR."
        context = ["The S2 observes the AO."]
        docs_by_text = {
            "The S2 observes the AO. It conducts ISR.": _FakeDoc([]),
            sentence: _FakeDoc(
                [
                    _FakeToken("It", pos_="PRON", dep_="nsubj", resolved_text="The S2"),
                    _FakeToken("conducts", pos_="VERB", dep_="ROOT"),
                    _FakeToken("ISR", pos_="NOUN", dep_="dobj"),
                    _FakeToken(".", pos_="PUNCT", dep_="punct"),
                ]
            ),
            "The S2 observes the AO.": _FakeDoc([], noun_chunks=["The S2"]),
        }
        docs_by_text[sentence]._tokens[1].children = [docs_by_text[sentence]._tokens[0]]
        fake_spacy = _FakeSpacyModule(docs_by_text)

        monkeypatch.setattr(normalize_module, "import_module", lambda name: fake_spacy if name == "spacy" else object())

        adapter = NLPAdapter()

        assert adapter.complete(sentence, context) == "The S2 conducts ISR."
        assert "coreferee" in fake_spacy.fake_nlp.added_pipes

    def test_sentence_with_no_subject_prepends_subject_from_context(self, monkeypatch):
        sentence = "Conducts ISR."
        context = ["The S2 observes the AO."]
        docs_by_text = {
            "The S2 observes the AO. Conducts ISR.": _FakeDoc([]),
            sentence: _FakeDoc(
                [
                    _FakeToken("Conducts", pos_="VERB", dep_="ROOT"),
                    _FakeToken("ISR", pos_="NOUN", dep_="dobj"),
                    _FakeToken(".", pos_="PUNCT", dep_="punct"),
                ]
            ),
            "The S2 observes the AO.": _FakeDoc([], noun_chunks=["The S2"]),
        }
        fake_spacy = _FakeSpacyModule(docs_by_text)

        monkeypatch.setattr(normalize_module, "import_module", lambda name: fake_spacy if name == "spacy" else object())

        adapter = NLPAdapter()

        assert adapter.complete(sentence, context) == "The S2 conducts ISR."

    def test_sentence_with_no_subject_and_no_context_returns_ambiguous(self, monkeypatch):
        sentence = "Conducts ISR."
        docs_by_text = {
            sentence: _FakeDoc(
                [
                    _FakeToken("Conducts", pos_="VERB", dep_="ROOT"),
                    _FakeToken("ISR", pos_="NOUN", dep_="dobj"),
                    _FakeToken(".", pos_="PUNCT", dep_="punct"),
                ]
            ),
        }
        fake_spacy = _FakeSpacyModule(docs_by_text)

        monkeypatch.setattr(normalize_module, "import_module", lambda name: fake_spacy if name == "spacy" else object())

        adapter = NLPAdapter()

        assert adapter.complete(sentence, []) == "[AMBIGUOUS] Conducts ISR."

    def test_original_vocabulary_is_preserved_except_resolution_and_subject_completion(self, monkeypatch):
        sentence = "It secures OBJ."
        context = ["The BN observes the AO."]
        docs_by_text = {
            "The BN observes the AO. It secures OBJ.": _FakeDoc([]),
            sentence: _FakeDoc(
                [
                    _FakeToken("It", pos_="PRON", dep_="nsubj", resolved_text="The BN"),
                    _FakeToken("secures", pos_="VERB", dep_="ROOT"),
                    _FakeToken("OBJ", pos_="NOUN", dep_="dobj"),
                    _FakeToken(".", pos_="PUNCT", dep_="punct"),
                ]
            ),
            "The BN observes the AO.": _FakeDoc([], noun_chunks=["The BN"]),
        }
        docs_by_text[sentence]._tokens[1].children = [docs_by_text[sentence]._tokens[0]]
        fake_spacy = _FakeSpacyModule(docs_by_text)

        monkeypatch.setattr(normalize_module, "import_module", lambda name: fake_spacy if name == "spacy" else object())

        adapter = NLPAdapter()

        assert adapter.complete(sentence, context) == "The BN secures OBJ."
