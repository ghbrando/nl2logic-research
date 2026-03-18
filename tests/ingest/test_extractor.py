from __future__ import annotations

import importlib
import shutil
from pathlib import Path

import pytest

from src.ingest.extractor import (
    DocSentence,
    ExtractionError,
    PageText,
    clean_line,
    clean_page_text,
    extract_pages,
    extract_sentences,
    is_section_heading,
    read_jsonl,
    split_sentences,
    write_jsonl,
)

extractor_module = importlib.import_module("src.ingest.extractor")


class _FakeSpan:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeDoc:
    def __init__(self, sentences: list[str]) -> None:
        self.sents = [_FakeSpan(sentence) for sentence in sentences]


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


class _FakePage:
    def __init__(self, *, text: str = "", error: Exception | None = None) -> None:
        self._text = text
        self._error = error

    def extract_text(self) -> str:
        if self._error is not None:
            raise self._error
        return self._text


class _FakePDF:
    def __init__(self, pages: list[_FakePage]) -> None:
        self.pages = pages

    def __enter__(self) -> _FakePDF:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class _FakePdfPlumberModule:
    def __init__(self, pages: list[_FakePage] | None = None, *, open_error: Exception | None = None) -> None:
        self._pages = pages or []
        self._open_error = open_error

    def open(self, path: Path) -> _FakePDF:
        if self._open_error is not None:
            raise self._open_error
        return _FakePDF(self._pages)


@pytest.fixture
def scratch_dir() -> Path:
    path = Path(".scratch_ingest")
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    path.mkdir(exist_ok=True)
    yield path
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


class TestCleanLine:
    def test_all_caps_short_line_returns_none(self):
        assert clean_line("INTELLIGENCE SUMMARY") is None

    def test_standalone_page_number_returns_none(self):
        assert clean_line("42") is None

    def test_fm_header_pattern_returns_none(self):
        assert clean_line("FM 2-0") is None

    def test_normal_sentence_returns_cleaned_string(self):
        assert clean_line("The battalion conducts reconnaissance operations.") == (
            "The battalion conducts reconnaissance operations."
        )

    def test_intra_line_whitespace_is_collapsed(self):
        assert clean_line("The\t battalion   conducts operations.") == "The battalion conducts operations."


class TestCleanPageText:
    def test_hyphenated_line_break_is_rejoined(self):
        raw = "The battalion conducts opera-\ntions to support the brigade."

        assert clean_page_text(raw) == "The battalion conducts operations to support the brigade."

    def test_short_all_caps_heading_is_preserved_even_if_clean_line_would_discard_it(self):
        raw = "TASKS\nThe battalion conducts reconnaissance operations."

        assert clean_page_text(raw) == "TASKS\nThe battalion conducts reconnaissance operations."

    def test_header_lines_removed_and_surrounding_lines_preserved(self):
        raw = "FM 2-0\nThe battalion collects intelligence.\n42\nThe brigade acts decisively."

        assert clean_page_text(raw) == "The battalion collects intelligence.\nThe brigade acts decisively."

    def test_consecutive_blank_lines_collapse_to_single_blank_line(self):
        raw = "The battalion collects intelligence.\n\n\nThe brigade acts decisively."

        assert clean_page_text(raw) == "The battalion collects intelligence.\n\nThe brigade acts decisively."

    def test_internal_newlines_are_preserved(self):
        raw = "The battalion collects intelligence.\nThe brigade acts decisively."

        assert clean_page_text(raw) == "The battalion collects intelligence.\nThe brigade acts decisively."

    def test_paragraph_boundary_is_preserved_between_paragraphs(self):
        raw = "The battalion collects intelligence.\n\nThe brigade acts decisively."

        assert clean_page_text(raw) == "The battalion collects intelligence.\n\nThe brigade acts decisively."


class TestIsSectionHeading:
    def test_all_caps_line_under_ten_words_returns_true(self):
        assert is_section_heading("COLLECTION PLAN") is True

    def test_title_case_line_under_ten_words_returns_true(self):
        assert is_section_heading("Collection Plan") is True

    def test_line_ending_with_period_returns_false(self):
        assert is_section_heading("Collection Plan.") is False

    def test_line_over_ten_words_returns_false(self):
        assert is_section_heading("This Heading Contains More Than Nine Distinct Words In Practice Today") is False

    def test_empty_line_returns_false(self):
        assert is_section_heading("") is False


class TestSplitSentences:
    def test_sentence_under_five_words_is_filtered(self, monkeypatch):
        fake_spacy = _FakeSpacyModule({"Input": _FakeDoc(["Too short here."])})
        monkeypatch.setattr(extractor_module, "import_module", lambda name: fake_spacy if name == "spacy" else object())

        assert split_sentences("Input") == []

    def test_bullet_style_line_is_filtered(self, monkeypatch):
        fake_spacy = _FakeSpacyModule({"Input": _FakeDoc(["1. Conduct ISR to support operations."])})
        monkeypatch.setattr(extractor_module, "import_module", lambda name: fake_spacy if name == "spacy" else object())

        assert split_sentences("Input") == []

    def test_unicode_bullet_line_is_filtered(self, monkeypatch):
        fake_spacy = _FakeSpacyModule({"Input": _FakeDoc(["⚫ Conduct ISR to support operations."])})
        monkeypatch.setattr(extractor_module, "import_module", lambda name: fake_spacy if name == "spacy" else object())

        assert split_sentences("Input") == []

    def test_clean_sentence_passes_through(self, monkeypatch):
        fake_spacy = _FakeSpacyModule(
            {"Input": _FakeDoc(["The battalion conducts reconnaissance operations to support the brigade."])}
        )
        monkeypatch.setattr(extractor_module, "import_module", lambda name: fake_spacy if name == "spacy" else object())

        assert split_sentences("Input") == ["The battalion conducts reconnaissance operations to support the brigade."]


class TestExtractPages:
    def test_returns_correct_page_text_list(self, monkeypatch, scratch_dir):
        pdf_path = scratch_dir / "fm2-0.pdf"
        pdf_path.write_text("placeholder", encoding="utf-8")
        fake_pdfplumber = _FakePdfPlumberModule(
            pages=[_FakePage(text="Page one."), _FakePage(text="Page two.")]
        )
        monkeypatch.setattr(
            extractor_module,
            "import_module",
            lambda name: fake_pdfplumber if name == "pdfplumber" else object(),
        )

        assert extract_pages(pdf_path) == [
            PageText(page_number=1, text="Page one."),
            PageText(page_number=2, text="Page two."),
        ]

    def test_raises_extraction_error_on_missing_file(self, scratch_dir):
        with pytest.raises(ExtractionError, match="PDF not found"):
            extract_pages(scratch_dir / "missing.pdf")

    def test_raises_extraction_error_on_pdf_open_failure(self, monkeypatch, scratch_dir):
        pdf_path = scratch_dir / "fm2-0.pdf"
        pdf_path.write_text("placeholder", encoding="utf-8")
        fake_pdfplumber = _FakePdfPlumberModule(open_error=RuntimeError("bad pdf"))
        monkeypatch.setattr(
            extractor_module,
            "import_module",
            lambda name: fake_pdfplumber if name == "pdfplumber" else object(),
        )

        with pytest.raises(ExtractionError, match="Failed to open PDF"):
            extract_pages(pdf_path)

    def test_per_page_failure_emits_warning_and_continues(self, monkeypatch, scratch_dir):
        pdf_path = scratch_dir / "fm2-0.pdf"
        pdf_path.write_text("placeholder", encoding="utf-8")
        fake_pdfplumber = _FakePdfPlumberModule(
            pages=[
                _FakePage(text="Page one."),
                _FakePage(error=RuntimeError("bad page")),
                _FakePage(text="Page three."),
            ]
        )
        monkeypatch.setattr(
            extractor_module,
            "import_module",
            lambda name: fake_pdfplumber if name == "pdfplumber" else object(),
        )

        with pytest.warns(UserWarning, match="Skipping page 2: bad page"):
            pages = extract_pages(pdf_path)

        assert pages == [
            PageText(page_number=1, text="Page one."),
            PageText(page_number=3, text="Page three."),
        ]


class TestExtractSentences:
    def test_all_caps_section_heading_is_attached_to_following_sentence(self, monkeypatch, scratch_dir):
        pdf_path = scratch_dir / "fm2-0.pdf"
        pdf_path.write_text("placeholder", encoding="utf-8")
        fake_pdfplumber = _FakePdfPlumberModule(
            pages=[
                _FakePage(
                    text="TASKS\nThe battalion conducts reconnaissance operations to support the brigade."
                )
            ]
        )
        docs_by_text = {
            "The battalion conducts reconnaissance operations to support the brigade.": _FakeDoc(
                ["The battalion conducts reconnaissance operations to support the brigade."]
            )
        }
        fake_spacy = _FakeSpacyModule(docs_by_text)

        def fake_import(name: str):
            if name == "pdfplumber":
                return fake_pdfplumber
            if name == "spacy":
                return fake_spacy
            raise AssertionError(f"Unexpected import: {name}")

        monkeypatch.setattr(extractor_module, "import_module", fake_import)

        sentences = extract_sentences(pdf_path)

        assert sentences == [
            DocSentence(
                sentence="The battalion conducts reconnaissance operations to support the brigade.",
                page_number=1,
                position=0,
                section="TASKS",
            )
        ]

    def test_duplicate_sentences_are_deduplicated_and_metadata_is_attached(self, monkeypatch, scratch_dir):
        pdf_path = scratch_dir / "fm2-0.pdf"
        pdf_path.write_text("placeholder", encoding="utf-8")
        fake_pdfplumber = _FakePdfPlumberModule(
            pages=[
                _FakePage(
                    text=(
                        "The battalion collects intelligence to support the commander.\n\n"
                        "Collection Plan\n"
                        "The battalion conducts reconnaissance operations to support the brigade."
                    )
                ),
                _FakePage(
                    text=(
                        "Collection Plan\n"
                        "The battalion conducts reconnaissance operations to support the brigade.\n"
                        "The brigade secures the objective area to support the division."
                    )
                ),
            ]
        )
        docs_by_text = {
            "The battalion collects intelligence to support the commander.": _FakeDoc(
                ["The battalion collects intelligence to support the commander."]
            ),
            "The battalion conducts reconnaissance operations to support the brigade.": _FakeDoc(
                ["The battalion conducts reconnaissance operations to support the brigade."]
            ),
            (
                "The battalion conducts reconnaissance operations to support the brigade.\n"
                "The brigade secures the objective area to support the division."
            ): _FakeDoc(
                [
                    "The battalion conducts reconnaissance operations to support the brigade.",
                    "The brigade secures the objective area to support the division.",
                ]
            ),
        }
        fake_spacy = _FakeSpacyModule(docs_by_text)

        def fake_import(name: str):
            if name == "pdfplumber":
                return fake_pdfplumber
            if name == "spacy":
                return fake_spacy
            raise AssertionError(f"Unexpected import: {name}")

        monkeypatch.setattr(extractor_module, "import_module", fake_import)

        sentences = extract_sentences(pdf_path)

        assert sentences == [
            DocSentence(
                sentence="The battalion collects intelligence to support the commander.",
                page_number=1,
                position=0,
                section="",
            ),
            DocSentence(
                sentence="The battalion conducts reconnaissance operations to support the brigade.",
                page_number=1,
                position=1,
                section="Collection Plan",
            ),
            DocSentence(
                sentence="The brigade secures the objective area to support the division.",
                page_number=2,
                position=1,
                section="Collection Plan",
            ),
        ]

    def test_per_page_warning_from_extract_pages_propagates(self, monkeypatch, scratch_dir):
        pdf_path = scratch_dir / "fm2-0.pdf"
        pdf_path.write_text("placeholder", encoding="utf-8")
        fake_pdfplumber = _FakePdfPlumberModule(
            pages=[
                _FakePage(text="Collection Plan\nThe battalion conducts reconnaissance operations to support the brigade."),
                _FakePage(error=RuntimeError("bad page")),
            ]
        )
        docs_by_text = {
            "The battalion conducts reconnaissance operations to support the brigade.": _FakeDoc(
                ["The battalion conducts reconnaissance operations to support the brigade."]
            )
        }
        fake_spacy = _FakeSpacyModule(docs_by_text)

        def fake_import(name: str):
            if name == "pdfplumber":
                return fake_pdfplumber
            if name == "spacy":
                return fake_spacy
            raise AssertionError(f"Unexpected import: {name}")

        monkeypatch.setattr(extractor_module, "import_module", fake_import)

        with pytest.warns(UserWarning, match="Skipping page 2: bad page"):
            sentences = extract_sentences(pdf_path)

        assert sentences == [
            DocSentence(
                sentence="The battalion conducts reconnaissance operations to support the brigade.",
                page_number=1,
                position=0,
                section="Collection Plan",
            )
        ]


class TestJsonlRoundTrip:
    def test_round_trip_preserves_all_fields(self, scratch_dir):
        path = scratch_dir / "sentences.jsonl"
        original = [
            DocSentence(
                sentence="The battalion conducts reconnaissance operations to support the brigade.",
                page_number=7,
                position=2,
                section="Collection Plan",
            )
        ]

        write_jsonl(original, path)

        assert read_jsonl(path) == original

    def test_missing_required_field_raises_extraction_error(self, scratch_dir):
        path = scratch_dir / "bad.jsonl"
        path.write_text('{"sentence":"x","page_number":1,"position":0}\n', encoding="utf-8")

        with pytest.raises(ExtractionError, match="Missing required field"):
            read_jsonl(path)
