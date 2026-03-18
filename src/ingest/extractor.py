"""PDF field-manual extraction for the NL2Logic ingest pipeline."""

from __future__ import annotations

import json
import re
import unicodedata
import warnings
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

_SPACY_MODEL = "en_core_web_lg"
_INLINE_WHITESPACE_RE = re.compile(r"[ \t]+")
_TRAILING_HYPHEN_RE = re.compile(r"\w-\s*$")
_WORD_START_RE = re.compile(r"^\w")
_PURE_NUMERIC_RE = re.compile(r"^\d+$")
_PURE_NUMERIC_OR_PUNCT_RE = re.compile(r"^[\d\W_]+$")
_BULLET_RE = re.compile("^\\s*(?:\\d+\\.|[A-Za-z]\\.|[-\u25aa\u26ab\u2022\u25e6\u25cf\u25a0])\\s*")
_FM_HEADER_PATTERNS = (
    re.compile(r"^FM\s+\d[\d-]*$", re.IGNORECASE),
    re.compile(r"^CHAPTER\s+\w+(?:\s+\w+)*$", re.IGNORECASE),
    re.compile(r"^SECTION\s+\w+(?:\s+\w+)*$", re.IGNORECASE),
    re.compile(
        r"^(?:\d{1,2}\s+)?(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}$",
        re.IGNORECASE,
    ),
)
_MIN_SENTENCE_WORDS = 5
_TITLE_CASE_STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "by",
    "for",
    "from",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}


@dataclass(frozen=True)
class PageText:
    page_number: int
    text: str


@dataclass(frozen=True)
class DocSentence:
    sentence: str
    page_number: int
    position: int
    section: str


class ExtractionError(Exception):
    """Raised when ingest extraction fails."""


def clean_line(line: str) -> str | None:
    """
    Clean a single line of PDF text while preserving line structure.

    Returns ``None`` for discarded headers/footers, ``""`` for blank lines, and
    the cleaned line otherwise.
    """
    normalized = unicodedata.normalize("NFKC", line).replace("\u00ad", "")
    normalized = re.sub(r"(?<=\w)-\s+(?=\w)", "", normalized)
    normalized = _INLINE_WHITESPACE_RE.sub(" ", normalized).strip()

    if not normalized:
        return ""
    if _PURE_NUMERIC_RE.fullmatch(normalized):
        return None
    if any(pattern.fullmatch(normalized) for pattern in _FM_HEADER_PATTERNS):
        return None

    words = normalized.split()
    if len(words) < 6 and any(character.isalpha() for character in normalized) and normalized.upper() == normalized:
        return None

    return normalized


def clean_page_text(raw_text: str) -> str:
    """Apply line-aware cleaning to a full PDF page while preserving newlines."""
    lines = raw_text.split("\n")
    repaired_lines: list[str] = []
    index = 0
    while index < len(lines):
        current = lines[index]
        if index + 1 < len(lines):
            next_line = lines[index + 1]
            if _TRAILING_HYPHEN_RE.search(current.rstrip()) and _WORD_START_RE.match(next_line.lstrip()):
                repaired_lines.append(current.rstrip()[:-1] + next_line.lstrip())
                index += 2
                continue
        repaired_lines.append(current)
        index += 1

    cleaned_lines: list[str] = []
    for line in repaired_lines:
        normalized = unicodedata.normalize("NFKC", line).replace("\u00ad", "")
        normalized = re.sub(r"(?<=\w)-\s+(?=\w)", "", normalized)
        normalized = _INLINE_WHITESPACE_RE.sub(" ", normalized).strip()

        looks_like_preservable_heading = (
            bool(normalized)
            and not _PURE_NUMERIC_RE.fullmatch(normalized)
            and not any(pattern.fullmatch(normalized) for pattern in _FM_HEADER_PATTERNS)
            and is_section_heading(normalized)
        )
        if looks_like_preservable_heading:
            cleaned = normalized
        else:
            cleaned = clean_line(line)
        if cleaned is None:
            continue
        if cleaned == "":
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            continue
        cleaned_lines.append(cleaned)

    while cleaned_lines and cleaned_lines[0] == "":
        cleaned_lines.pop(0)
    while cleaned_lines and cleaned_lines[-1] == "":
        cleaned_lines.pop()

    return "\n".join(cleaned_lines)


def _is_title_case_heading(line: str) -> bool:
    words = line.split()
    if not words:
        return False

    for word in words:
        stripped = word.strip("()[]{}:,;/-")
        if not stripped or stripped.isupper() or stripped.isdigit():
            continue
        if stripped.casefold() in _TITLE_CASE_STOPWORDS:
            continue
        if not stripped[0].isupper():
            return False
    return True


def is_section_heading(line: str) -> bool:
    """Return True when a cleaned line looks like a section heading."""
    stripped = line.strip()
    if not stripped or stripped.endswith("."):
        return False

    words = stripped.split()
    if len(words) >= 10:
        return False

    return stripped.upper() == stripped or _is_title_case_heading(stripped)


def _load_spacy_model():
    try:
        spacy = import_module("spacy")
        nlp = spacy.load(_SPACY_MODEL)
        pipe_names = list(getattr(nlp, "pipe_names", []))
        if "sentencizer" not in pipe_names and "parser" not in pipe_names and hasattr(nlp, "add_pipe"):
            nlp.add_pipe("sentencizer")
        return nlp
    except Exception as exc:
        raise ExtractionError("spaCy sentence splitter unavailable for ingest extraction.") from exc


def _split_sentences_with_nlp(text: str, nlp) -> list[str]:
    if not text.strip():
        return []

    try:
        doc = nlp(text)
    except Exception as exc:
        raise ExtractionError("spaCy sentence splitting failed during ingest extraction.") from exc

    sentences: list[str] = []
    for span in getattr(doc, "sents", []):
        sentence = _INLINE_WHITESPACE_RE.sub(" ", getattr(span, "text", str(span))).strip()
        if not sentence:
            continue
        if len(sentence.split()) < _MIN_SENTENCE_WORDS:
            continue
        if _PURE_NUMERIC_OR_PUNCT_RE.fullmatch(sentence):
            continue
        if _BULLET_RE.match(sentence):
            continue
        sentences.append(sentence)

    return sentences


def split_sentences(text: str) -> list[str]:
    """Split cleaned page text into filtered sentence strings."""
    return _split_sentences_with_nlp(text, _load_spacy_model())


def extract_pages(pdf_path: str | Path) -> list[PageText]:
    """Extract raw text from a PDF page-by-page."""
    path = Path(pdf_path)
    if not path.exists():
        raise ExtractionError(f"PDF not found: {path}")

    try:
        pdfplumber = import_module("pdfplumber")
    except Exception as exc:
        raise ExtractionError("pdfplumber is unavailable for ingest extraction.") from exc

    try:
        with pdfplumber.open(path) as pdf:
            pages: list[PageText] = []
            for index, page in enumerate(getattr(pdf, "pages", []), start=1):
                try:
                    text = page.extract_text() or ""
                except Exception as exc:
                    warnings.warn(f"Skipping page {index}: {exc}", UserWarning, stacklevel=2)
                    continue
                pages.append(PageText(page_number=index, text=text))
            return pages
    except ExtractionError:
        raise
    except Exception as exc:
        raise ExtractionError(f"Failed to open PDF: {path}") from exc


def _section_blocks(cleaned_text: str) -> list[tuple[str, str]]:
    section = ""
    current_lines: list[str] = []
    blocks: list[tuple[str, str]] = []

    for line in cleaned_text.splitlines():
        stripped = line.strip()
        if not stripped:
            if current_lines:
                blocks.append((section, "\n".join(current_lines)))
                current_lines = []
            continue
        if is_section_heading(stripped):
            if current_lines:
                blocks.append((section, "\n".join(current_lines)))
                current_lines = []
            section = stripped
            continue
        current_lines.append(stripped)

    if current_lines:
        blocks.append((section, "\n".join(current_lines)))

    return blocks


def extract_sentences(pdf_path: str | Path) -> list[DocSentence]:
    """Extract deduplicated sentence-level text plus source metadata from a PDF."""
    pages = extract_pages(pdf_path)
    nlp = _load_spacy_model()
    seen: set[str] = set()
    sentences: list[DocSentence] = []

    for page in pages:
        cleaned_text = clean_page_text(page.text)
        page_position = 0
        for section, block in _section_blocks(cleaned_text):
            for sentence in _split_sentences_with_nlp(block, nlp):
                position = page_position
                page_position += 1
                if sentence in seen:
                    continue
                seen.add(sentence)
                sentences.append(
                    DocSentence(
                        sentence=sentence,
                        page_number=page.page_number,
                        position=position,
                        section=section,
                    )
                )

    return sentences


def write_jsonl(sentences: list[DocSentence], output_path: str | Path) -> None:
    """Write extracted sentences to JSONL."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for sentence in sentences:
            handle.write(
                json.dumps(
                    {
                        "sentence": sentence.sentence,
                        "page_number": sentence.page_number,
                        "position": sentence.position,
                        "section": sentence.section,
                    }
                )
            )
            handle.write("\n")


def read_jsonl(input_path: str | Path) -> list[DocSentence]:
    """Read extracted sentences from JSONL."""
    path = Path(input_path)
    try:
        with open(path, encoding="utf-8") as handle:
            sentences: list[DocSentence] = []
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ExtractionError(f"Malformed JSONL at line {line_number} in {path}") from exc

                required = {"sentence", "page_number", "position", "section"}
                if not required.issubset(payload):
                    missing = ", ".join(sorted(required - set(payload)))
                    raise ExtractionError(f"Missing required field(s) at line {line_number} in {path}: {missing}")

                sentences.append(
                    DocSentence(
                        sentence=str(payload["sentence"]),
                        page_number=int(payload["page_number"]),
                        position=int(payload["position"]),
                        section=str(payload["section"]),
                    )
                )
            return sentences
    except ExtractionError:
        raise
    except OSError as exc:
        raise ExtractionError(f"Failed to read JSONL: {path}") from exc

