"""Sentence normalization adapters for NL2Logic preprocessing."""

from __future__ import annotations

import re
from dataclasses import dataclass
from importlib import import_module
from typing import Callable

_ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
_SPACY_MODEL = "en_core_web_lg"
_AMBIGUOUS_PREFIX = "[AMBIGUOUS]"
_WHITESPACE_RE = re.compile(r"\s+")
_VERB_POS = {"VERB", "AUX"}
_SUBJECT_DEPS = {"nsubj", "nsubjpass"}
_PRONOUN_POS = {"PRON"}
_PRONOUN_TEXTS = {
    "he",
    "him",
    "his",
    "she",
    "her",
    "hers",
    "it",
    "its",
    "they",
    "them",
    "their",
    "theirs",
    "this",
    "that",
    "these",
    "those",
}
_PUNCT_NO_SPACE = {".", ",", ";", ":", "!", "?", "%", ")", "]", "}"}


@dataclass(frozen=True)
class NormalizationResult:
    normalized: list[str]
    ambiguous: list[str]
    original: str


class NormalizationError(Exception):
    """Raised on transport failure or missing NLP model dependencies."""


def _clean_text(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text.strip())


def build_prompt(sentence: str, context: list[str] | None = None) -> str:
    context_items = [_clean_text(item) for item in (context or []) if _clean_text(item)]
    context_block = "\n".join(f"{index + 1}. {item}" for index, item in enumerate(context_items))
    if not context_block:
        context_block = "(none)"

    sentence_text = _clean_text(sentence)
    return (
        "You normalize military doctrine sentences for a downstream logic pipeline.\n"
        "Rewrite only the source sentence using the context window when needed.\n\n"
        "Context window:\n"
        f"{context_block}\n\n"
        "Source sentence:\n"
        f"{sentence_text}\n\n"
        "Instructions:\n"
        "- Replace all pronouns with their explicit referents from the context window.\n"
        "- Make all implicit subjects explicit.\n"
        "- No paraphrasing except explicit referent substitution and subject completion.\n"
        "- Preserve all other original vocabulary exactly.\n"
        "- Never add facts or terms not present in the source sentence or context window.\n"
        "- If a referent is genuinely ambiguous, output the sentence prefixed [AMBIGUOUS].\n"
        "- Return only rewritten sentences, one per line, with no explanation.\n"
    )


def parse_response(response_text: str, original: str) -> NormalizationResult:
    lines = [_clean_text(line) for line in response_text.splitlines() if _clean_text(line)]
    if not lines:
        return NormalizationResult(normalized=[], ambiguous=[original], original=original)

    normalized: list[str] = []
    ambiguous: list[str] = []
    for line in lines:
        if line.startswith(_AMBIGUOUS_PREFIX):
            content = _clean_text(line[len(_AMBIGUOUS_PREFIX) :]) or original
            ambiguous.append(content)
        else:
            normalized.append(line)

    if not normalized and not ambiguous:
        ambiguous.append(original)

    return NormalizationResult(normalized=normalized, ambiguous=ambiguous, original=original)


class AnthropicAdapter:
    def __init__(self, api_key: str, model: str = _ANTHROPIC_MODEL):
        self._api_key = api_key
        self._model = model
        try:
            anthropic = import_module("anthropic")
            self._client = anthropic.Anthropic(api_key=api_key)
        except Exception as exc:
            raise NormalizationError("Anthropic client unavailable for normalization.") from exc

    def __call__(self, sentence: str, context: list[str] | None = None) -> str:
        return self.complete(sentence, context or [])

    def complete(self, sentence: str, context: list[str] | None = None) -> str:
        prompt = build_prompt(sentence, context or [])
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=512,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:
            raise NormalizationError("Anthropic normalization request failed.") from exc

        blocks = getattr(response, "content", [])
        parts: list[str] = []
        for block in blocks:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "\n".join(parts).strip()


class NLPAdapter:
    def __init__(self):
        try:
            spacy = import_module("spacy")
            import_module("coreferee")
            self._nlp = spacy.load(_SPACY_MODEL)
            if hasattr(self._nlp, "pipe_names") and "coreferee" not in self._nlp.pipe_names:
                self._nlp.add_pipe("coreferee")
        except Exception as exc:
            raise NormalizationError(
                "spaCy normalization backend unavailable. Install en_core_web_lg and coreferee."
            ) from exc

    def __call__(self, sentence: str, context: list[str] | None = None) -> str:
        return self.complete(sentence, context or [])

    def complete(self, sentence: str, context: list[str] | None = None) -> str:
        sentence_text = _clean_text(sentence)
        context_items = [_clean_text(item) for item in (context or []) if _clean_text(item)]
        if not sentence_text:
            return ""

        try:
            passage_text = " ".join([*context_items, sentence_text]).strip()
            passage_doc = self._nlp(passage_text)
            sentence_doc = self._nlp(sentence_text)
            context_doc = self._nlp(" ".join(context_items)) if context_items else None
        except Exception as exc:
            raise NormalizationError("spaCy normalization failed during parsing.") from exc

        resolved_sentence = self._resolve_sentence(sentence_doc, passage_doc, context_doc)
        if self._sentence_has_subject(sentence_doc):
            return resolved_sentence

        subject = self._infer_subject(context_doc)
        if subject is None:
            return f"{_AMBIGUOUS_PREFIX} {resolved_sentence}"

        completed = self._prepend_subject(subject, resolved_sentence)
        return completed

    def _resolve_sentence(self, sentence_doc, passage_doc, context_doc) -> str:
        tokens: list[str] = []
        for token in sentence_doc:
            replacement = None
            if self._is_pronoun(token):
                replacement = self._resolve_pronoun(token, passage_doc, context_doc)
            tokens.append(replacement or getattr(token, "text", str(token)))
        return self._join_tokens(tokens)

    @staticmethod
    def _is_pronoun(token) -> bool:
        text = getattr(token, "text", "")
        pos = getattr(token, "pos_", "")
        return pos in _PRONOUN_POS or text.casefold() in _PRONOUN_TEXTS

    def _resolve_pronoun(self, token, passage_doc, context_doc) -> str | None:
        token_extension = getattr(token, "_", None)
        for attribute in ("resolved_text", "coref_resolved", "resolved"):
            value = getattr(token_extension, attribute, None) if token_extension is not None else None
            if isinstance(value, str) and _clean_text(value):
                return _clean_text(value)

        for owner in (token_extension, getattr(passage_doc, "_", None), getattr(context_doc, "_", None)):
            chains = getattr(owner, "coref_chains", None)
            if chains is None or not hasattr(chains, "resolve"):
                continue
            try:
                resolved = chains.resolve(token)
            except Exception:
                continue
            resolved_text = self._coerce_resolved_text(resolved)
            if resolved_text:
                return resolved_text

        return self._infer_subject(context_doc)

    @staticmethod
    def _coerce_resolved_text(resolved) -> str | None:
        if isinstance(resolved, str):
            return _clean_text(resolved)
        if hasattr(resolved, "text"):
            return _clean_text(getattr(resolved, "text"))
        if isinstance(resolved, (list, tuple)):
            parts = []
            for item in resolved:
                if hasattr(item, "text"):
                    parts.append(getattr(item, "text"))
                elif isinstance(item, str):
                    parts.append(item)
            combined = _clean_text(" ".join(parts))
            return combined or None
        return None

    @staticmethod
    def _sentence_has_subject(sentence_doc) -> bool:
        root = None
        for token in sentence_doc:
            if getattr(token, "dep_", "") == "ROOT":
                root = token
                break

        if root is None or getattr(root, "pos_", "") not in _VERB_POS:
            return any(getattr(token, "dep_", "") in _SUBJECT_DEPS for token in sentence_doc)

        children = getattr(root, "children", [])
        return any(getattr(child, "dep_", "") in _SUBJECT_DEPS for child in children)

    @staticmethod
    def _infer_subject(context_doc) -> str | None:
        if context_doc is None:
            return None

        noun_chunks = list(getattr(context_doc, "noun_chunks", []))
        if noun_chunks:
            chunk_text = _clean_text(getattr(noun_chunks[-1], "text", ""))
            if chunk_text:
                return chunk_text

        for token in reversed(list(context_doc)):
            if getattr(token, "pos_", "") in {"NOUN", "PROPN"}:
                text = _clean_text(getattr(token, "text", ""))
                if text:
                    return text
        return None

    @staticmethod
    def _prepend_subject(subject: str, sentence: str) -> str:
        clean_subject = _clean_text(subject)
        clean_sentence = sentence.strip()
        if not clean_subject:
            return clean_sentence
        if clean_sentence.casefold().startswith(clean_subject.casefold()):
            return clean_sentence
        if clean_sentence and clean_sentence[0].isupper():
            clean_sentence = clean_sentence[0].lower() + clean_sentence[1:]
        return f"{clean_subject} {clean_sentence}"

    @staticmethod
    def _join_tokens(tokens: list[str]) -> str:
        text = ""
        for token in tokens:
            if not token:
                continue
            if not text:
                text = token
                continue
            if token in _PUNCT_NO_SPACE or token.startswith("'"):
                text += token
            elif text.endswith(("(", "[", "{")):
                text += token
            else:
                text += f" {token}"
        return text


def normalize(
    sentence: str,
    context: list[str] | None = None,
    provider: Callable[[str, list[str]], str] | None = None,
) -> NormalizationResult:
    if provider is None:
        raise ValueError("normalize() requires a provider callable.")

    context_items = [_clean_text(item) for item in (context or []) if _clean_text(item)]
    try:
        response_text = provider(sentence, context_items)
    except NormalizationError:
        return NormalizationResult(normalized=[], ambiguous=[sentence], original=sentence)

    return parse_response(response_text, sentence)
