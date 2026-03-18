"""Archived LLM-based structured extraction baseline.

Instead of asking a small seq2seq model to generate CNL directly, this module
uses an LLM to *extract* structured claims from doctrine sentences, then maps
them deterministically to SUMO terms and assembles KIF.

Architecture:
    sentence → [LLM: structured extraction] → ExtractionResult
             → [SUMO mapper] → MappedClaim
             → [KIF assembler] → cnl + kif strings
             → [grounding validator] → accept / review / reject
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Sequence

_LOGGER = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CLASSES_PATH = _REPO_ROOT / "data" / "training_pairs" / "sumo_classes.jsonl"
_RELATIONS_PATH = _REPO_ROOT / "data" / "training_pairs" / "sumo_relations.jsonl"
_DOCTRINE_DOMAIN_PATH = _REPO_ROOT / "data" / "ontology" / "doctrine_domain.kif"

_ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
_DEFAULT_LOCAL_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

_WHITESPACE_RE = re.compile(r"\s+")
_PASCAL_BOUNDARY_RE = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

# ── Data structures ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ExtractedClaim:
    """A single structured claim extracted by the LLM."""

    pattern: str  # instance | subclass | relation | conditional | existential
    subject: str  # noun phrase from source
    predicate: str  # verb/relationship from source (empty for instance/subclass)
    object: str  # noun phrase from source (empty for instance)
    quantifier: str  # universal | existential | none
    negated: bool
    source_span: str  # substring from source justifying this claim
    confidence: str  # high | medium | low


@dataclass(frozen=True)
class ExtractionResult:
    """Full extraction result for one sentence."""

    formalizable: bool
    claims: list[ExtractedClaim]
    reason: str  # why not formalizable, or empty


@dataclass(frozen=True)
class MappedClaim:
    """A claim with SUMO terms resolved."""

    pattern: str
    subject_class: str | None  # SUMO class name
    relation: str | None  # SUMO relation name
    object_class: str | None  # SUMO class name
    quantifier: str
    negated: bool
    source_span: str
    confidence: str
    # Original extracted phrases for provenance
    subject_phrase: str
    predicate_phrase: str
    object_phrase: str


@dataclass(frozen=True)
class AssembledFact:
    """A fully assembled CNL + KIF fact ready for grounding validation."""

    cnl: str
    kif: str
    pattern: str
    relation: str | None
    terms: list[str]
    source_span: str
    confidence: str


# ── Extraction prompt ───────────────────────────────────────────────────────

_EXTRACTION_SYSTEM_PROMPT = """\
You are a precise information extraction system for military doctrine documents.
Your job is to identify formalizable factual claims in doctrine sentences and
extract their structured components.

A sentence is formalizable if it expresses one or more of these patterns:
- INSTANCE: "X is a Y" / "X is a type of entity called Y"
- SUBCLASS: "X is a kind/type of Y" / "Every X is a Y" (taxonomic)
- RELATION: "X has/requires/supports/provides Y" / "X is related to Y by R"
- CONDITIONAL: "If X then Y" / "Every X must/should Y" / "When X, Y"
- EXISTENTIAL: "There exists X" / "Some X exists" / "There is at least one X"

A sentence is NOT formalizable if it:
- Uses unresolved pronouns (it, they, them, this) without clear referents
- Is procedural ("do X, then do Y") without clear factual content
- Is purely descriptive atmosphere or commentary
- References figures, tables, appendices, or page numbers
- Is a definition that requires multiple complex clauses to express
- Contains vague modifiers without concrete relationships ("is important", "is key")

CRITICAL RULES:
- Extract ONLY what the sentence actually says. Never infer or add information.
- subject and object must be EXACT noun phrases from the source sentence.
- predicate must be a verb or relationship phrase from the source sentence.
- source_span must be a substring of the original sentence.
- If confidence is low, still extract but mark it as low.
- Prefer fewer, higher-quality claims over many weak claims.
- One claim per distinct factual assertion in the sentence.
"""

_EXTRACTION_USER_TEMPLATE = """\
Extract structured claims from this doctrine sentence.

Sentence: {sentence}

Respond with valid JSON only. Use this exact schema:
{{
  "formalizable": true or false,
  "reason": "why not formalizable (empty string if formalizable)",
  "claims": [
    {{
      "pattern": "instance|subclass|relation|conditional|existential",
      "subject": "exact noun phrase from sentence",
      "predicate": "verb or relationship from sentence",
      "object": "exact noun phrase from sentence",
      "quantifier": "universal|existential|none",
      "negated": false,
      "source_span": "substring of sentence justifying this claim",
      "confidence": "high|medium|low"
    }}
  ]
}}

If not formalizable, set claims to an empty array.
Respond with the JSON object only, no markdown fencing or explanation."""


# ── LLM client ──────────────────────────────────────────────────────────────


class ExtractionError(Exception):
    """Raised on LLM transport or parse failure."""


class StructuredExtractor:
    """Extracts structured claims from doctrine sentences using an LLM.

    Supports two backends:
        - "local" (default): HuggingFace model running on GPU (e.g. Mistral-7B)
        - "anthropic": Claude API (requires api_key)
    """

    def __init__(
        self,
        *,
        backend: str = "local",
        model: str | None = None,
        api_key: str | None = None,
        device: str | None = None,
    ) -> None:
        self._backend = backend
        if backend == "anthropic":
            self._model_name = model or _ANTHROPIC_MODEL
            if not api_key:
                raise ExtractionError("Anthropic backend requires api_key.")
            try:
                anthropic = import_module("anthropic")
                self._client = anthropic.Anthropic(api_key=api_key)
            except Exception as exc:
                raise ExtractionError("Anthropic client unavailable.") from exc
        elif backend == "local":
            self._model_name = model or _DEFAULT_LOCAL_MODEL
            self._load_local_model(device)
        else:
            raise ExtractionError(f"Unknown backend: {backend}")

    def _load_local_model(self, device: str | None) -> None:
        """Load a HuggingFace causal LM for local inference."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ExtractionError(
                "Local backend requires torch and transformers. "
                "Install with: pip install torch transformers"
            ) from exc

        _LOGGER.info("Loading local model: %s", self._model_name)

        if device is None:
            import torch as _torch
            device = "cuda" if _torch.cuda.is_available() else "cpu"

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name, trust_remote_code=True
        )
        self._local_model = AutoModelForCausalLM.from_pretrained(
            self._model_name,
            torch_dtype="auto",
            device_map=device if device != "cpu" else None,
            trust_remote_code=True,
        )
        if device == "cpu":
            self._local_model = self._local_model.to("cpu")

        # Ensure pad token is set
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._device = device
        _LOGGER.info("Model loaded on %s", device)

    def extract(self, sentence: str) -> ExtractionResult:
        """Extract structured claims from a single doctrine sentence."""
        if self._backend == "anthropic":
            return self._extract_anthropic(sentence)
        return self._extract_local(sentence)

    def _extract_anthropic(self, sentence: str) -> ExtractionResult:
        """Extract using Anthropic API."""
        user_prompt = _EXTRACTION_USER_TEMPLATE.format(sentence=sentence.strip())
        try:
            response = self._client.messages.create(
                model=self._model_name,
                max_tokens=1024,
                temperature=0,
                system=_EXTRACTION_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
        except Exception as exc:
            raise ExtractionError(f"Anthropic request failed: {exc}") from exc

        text = ""
        for block in getattr(response, "content", []):
            t = getattr(block, "text", None)
            if t:
                text += t

        return self._parse_response(text.strip())

    def _extract_local(self, sentence: str) -> ExtractionResult:
        """Extract using a local HuggingFace model."""
        import torch

        user_prompt = _EXTRACTION_USER_TEMPLATE.format(sentence=sentence.strip())

        # Build chat messages
        messages = [
            {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        # Apply chat template
        try:
            prompt_text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Fallback for models without chat templates
            prompt_text = (
                f"<s>[INST] {_EXTRACTION_SYSTEM_PROMPT}\n\n{user_prompt} [/INST]"
            )

        inputs = self._tokenizer(
            prompt_text, return_tensors="pt", truncation=True, max_length=2048
        )
        inputs = {k: v.to(self._local_model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._local_model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.1,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        # Decode only the generated tokens (exclude prompt)
        generated = outputs[0][inputs["input_ids"].shape[1] :]
        text = self._tokenizer.decode(generated, skip_special_tokens=True).strip()

        return self._parse_response(text)

    def extract_batch(self, sentences: Sequence[str]) -> list[ExtractionResult]:
        """Extract claims from multiple sentences (sequential)."""
        results: list[ExtractionResult] = []
        for sentence in sentences:
            try:
                results.append(self.extract(sentence))
            except ExtractionError as exc:
                _LOGGER.warning("Extraction failed for sentence: %s — %s", sentence[:80], exc)
                results.append(
                    ExtractionResult(formalizable=False, claims=[], reason=f"extraction_error: {exc}")
                )
        return results

    @staticmethod
    def _parse_response(text: str) -> ExtractionResult:
        """Parse LLM JSON response into ExtractionResult."""
        # Strip markdown fencing if present
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines).strip()

        # Try to extract JSON object if there's surrounding text
        json_match = re.search(r"\{[\s\S]*\}", cleaned)
        if json_match:
            cleaned = json_match.group(0)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ExtractionError(
                f"LLM response is not valid JSON: {exc}\nResponse: {text[:300]}"
            ) from exc

        formalizable = bool(data.get("formalizable", False))
        reason = str(data.get("reason", ""))
        claims: list[ExtractedClaim] = []

        for claim_data in data.get("claims", []):
            try:
                claims.append(
                    ExtractedClaim(
                        pattern=str(claim_data.get("pattern", "relation")),
                        subject=str(claim_data.get("subject", "")),
                        predicate=str(claim_data.get("predicate", "")),
                        object=str(claim_data.get("object", "")),
                        quantifier=str(claim_data.get("quantifier", "none")),
                        negated=bool(claim_data.get("negated", False)),
                        source_span=str(claim_data.get("source_span", "")),
                        confidence=str(claim_data.get("confidence", "medium")),
                    )
                )
            except Exception as exc:
                _LOGGER.warning("Skipping malformed claim: %s — %s", claim_data, exc)

        return ExtractionResult(formalizable=formalizable, claims=claims, reason=reason)


# ── SUMO term mapper ───────────────────────────────────────────────────────


def _normalise_phrase(text: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation edges."""
    text = _PASCAL_BOUNDARY_RE.sub(" ", text.replace("_", " "))
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return _WHITESPACE_RE.sub(" ", text).strip()


def _load_class_terms(path: Path) -> dict[str, str]:
    """Load SUMO classes → {normalized_phrase: TermName}."""
    lookup: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            term = json.loads(line)["term"]
            normalized = _normalise_phrase(term)
            lookup[normalized] = term
    return lookup


def _load_relation_terms(path: Path) -> dict[str, dict]:
    """Load SUMO relations → {normalized_phrase: {term, arity, signature}}."""
    lookup: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            term = data["term"]
            normalized = _normalise_phrase(term)
            lookup[normalized] = {
                "term": term,
                "arity": data.get("arity", 2),
                "signature": data.get("signature", {}),
            }
    return lookup


# Common doctrine verbs/phrases → SUMO relation mappings
_VERB_TO_RELATION: dict[str, str] = {
    "supports": "supports",
    "support": "supports",
    "requires": "requires",
    "require": "requires",
    "provides": "provides",
    "provide": "provides",
    "includes": "includes",
    "include": "includes",
    "contains": "contains",
    "contain": "contains",
    "uses": "uses",
    "use": "uses",
    "enables": "enables",
    "enable": "enables",
    "prevents": "prevents",
    "prevent": "prevents",
    "causes": "causes",
    "cause": "causes",
    "precedes": "precedes",
    "precede": "precedes",
    "follows": "follows",
    "follow": "follows",
    "has part": "part",
    "is part of": "part",
    "is a member of": "member",
    "belongs to": "member",
    "commands": "commandingOfficer",
    "controls": "controls",
    "employs": "employs",
    "conducts": "agent",
    "performs": "agent",
    "executes": "agent",
    "facilitates": "causes",
    "contributes to": "causes",
}


# ── Doctrine-specific phrase → SUMO class mappings ────────────────────────
# These are checked FIRST (highest priority) so that multi-word doctrine
# phrases map to the correct doctrine-domain class instead of a generic
# SUMO substring match.  Keys are lowercase NL phrases.

_DOCTRINE_PHRASE_TO_CLASS: dict[str, str] = {
    # Intelligence concepts
    "intelligence process": "IntelligenceProcess",
    "intelligence product": "IntelligenceProduct",
    "intelligence enterprise": "IntelligenceEnterprise",
    "intelligence professional": "IntelligenceProfessional",
    "intelligence professionals": "IntelligenceProfessional",
    "army intelligence professional": "IntelligenceProfessional",
    "army intelligence professionals": "IntelligenceProfessional",
    "intelligence discipline": "IntelligenceDiscipline",
    "combat information": "CombatInformation",
    "intelligence activities": "IntelligenceActivities",
    "military intelligence": "MilitaryIntelligence",
    # Warfighting function concepts
    "warfighting function": "WarfightingFunction",
    "intelligence warfighting function": "IntelligenceWarfightingFunction",
    # Operational environment and threat
    "operational environment": "OperationalEnvironment",
    "threat course of action": "ThreatCourseOfAction",
    "course of action": "ThreatCourseOfAction",
    # Doctrine and planning
    "doctrinal task": "DoctrinalTask",
    "intelligence warfighting function task": "IntelligenceWarfightingFunctionTask",
    # Collection and dissemination
    "information collection": "InformationCollection",
    "intelligence dissemination": "IntelligenceDissemination",
    # Command relationships
    "tactical commander": "TacticalCommander",
    "military commander": "MilitaryCommander",
    "combatant commander": "CombatantCommander",
    "joint forces commander": "JointForcesCommander",
    # Existing SUMO military terms worth surfacing
    "military organization": "MilitaryOrganization",
    "military process": "MilitaryProcess",
    "military person": "MilitaryPerson",
    "military force": "MilitaryForce",
    "military operation": "MilitaryOperation",
    "intelligence officer": "IntelligenceOfficer",
}


def _load_doctrine_domain_classes(path: Path) -> dict[str, str]:
    """Parse (subclass X Y) forms from doctrine_domain.kif → {normalized: TermName}."""
    result: dict[str, str] = {}
    if not path.exists():
        return result
    subclass_re = re.compile(r"^\(subclass\s+(\S+)\s+\S+\)")
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = subclass_re.match(line.strip())
            if m:
                term = m.group(1)
                normalized = _normalise_phrase(term)
                result[normalized] = term
    return result


class SUMOMapper:
    """Maps extracted noun phrases and predicates to SUMO ontology terms."""

    def __init__(
        self,
        *,
        classes_path: Path = _CLASSES_PATH,
        relations_path: Path = _RELATIONS_PATH,
        doctrine_domain_path: Path = _DOCTRINE_DOMAIN_PATH,
    ) -> None:
        self._classes = _load_class_terms(classes_path)
        self._relations = _load_relation_terms(relations_path)

        # Load doctrine-domain classes and merge (they take precedence)
        doctrine_classes = _load_doctrine_domain_classes(doctrine_domain_path)
        self._classes.update(doctrine_classes)

        # Build the doctrine phrase table (normalized keys → class names)
        self._doctrine_phrases = {
            _normalise_phrase(phrase): cls
            for phrase, cls in _DOCTRINE_PHRASE_TO_CLASS.items()
        }
        # Sort by phrase length descending so longer phrases match first
        self._doctrine_phrases_by_length = sorted(
            self._doctrine_phrases.items(), key=lambda kv: len(kv[0]), reverse=True
        )

        # Build reverse lookups for fast substring matching
        self._class_terms_by_length = sorted(
            self._classes.items(), key=lambda kv: len(kv[0]), reverse=True
        )
        self._relation_terms_by_length = sorted(
            self._relations.items(), key=lambda kv: len(kv[0]), reverse=True
        )

    @staticmethod
    def _word_boundary_match(key: str, text: str) -> bool:
        """Check if key appears in text at word boundaries."""
        pattern = rf"(?<![a-z0-9]){re.escape(key)}(?![a-z0-9])"
        return re.search(pattern, text) is not None

    def map_class(self, phrase: str) -> str | None:
        """Map a noun phrase to a SUMO class name, or None if no match.

        Priority order:
        1. Doctrine phrase table (longest match first) — catches multi-word
           doctrine terms like "intelligence warfighting function"
        2. Exact match in SUMO class vocabulary
        3. Head-noun preference — try the last N words, then N-1, etc.
           (English NPs are head-final: "Army intelligence professionals"
           → try "intelligence professionals" → match)
        4. Longest word-boundary substring match (fallback)
        """
        if not phrase.strip():
            return None
        normalized = _normalise_phrase(phrase)

        # 1. Doctrine phrase table (longest match first)
        for key, cls in self._doctrine_phrases_by_length:
            if self._word_boundary_match(key, normalized):
                return cls

        # 2. Exact match in SUMO classes
        if normalized in self._classes:
            return self._classes[normalized]

        # 3. Head-noun preference: try progressively shorter right-aligned
        #    subphrases (e.g. "army intelligence professionals" → try
        #    "intelligence professionals", then "professionals")
        words = normalized.split()
        if len(words) > 1:
            for start in range(1, len(words)):
                subphrase = " ".join(words[start:])
                # Check doctrine phrases first
                for key, cls in self._doctrine_phrases_by_length:
                    if self._word_boundary_match(key, subphrase):
                        return cls
                # Then exact SUMO match
                if subphrase in self._classes:
                    return self._classes[subphrase]

        # 4. Longest word-boundary match (min 4 chars to avoid spurious hits)
        for key, term in self._class_terms_by_length:
            if len(key) >= 4 and self._word_boundary_match(key, normalized):
                return term

        return None

    def map_relation(self, predicate: str) -> str | None:
        """Map a verb/predicate phrase to a SUMO relation name."""
        if not predicate.strip():
            return None
        normalized = _normalise_phrase(predicate)

        # Check verb-to-relation table first
        if normalized in _VERB_TO_RELATION:
            rel_name = _VERB_TO_RELATION[normalized]
            # Verify it's a real SUMO relation
            rel_norm = _normalise_phrase(rel_name)
            if rel_norm in self._relations:
                return self._relations[rel_norm]["term"]
            return rel_name  # Use mapping even if not in SUMO vocab

        # Exact match in SUMO relations
        if normalized in self._relations:
            return self._relations[normalized]["term"]

        # Word-boundary match (min 5 chars for relations)
        for key, data in self._relation_terms_by_length:
            if len(key) >= 5 and self._word_boundary_match(key, normalized):
                return data["term"]

        return None

    def map_claim(self, claim: ExtractedClaim) -> MappedClaim | None:
        """Map an extracted claim to SUMO terms. Returns None if unmappable."""
        subject_class = self.map_class(claim.subject)
        object_class = self.map_class(claim.object) if claim.object else None
        relation = self.map_relation(claim.predicate) if claim.predicate else None
        pattern = claim.pattern.lower()

        # For instance patterns: upgrade to subclass when both subject and
        # object map to valid SUMO classes (e.g. "Combat information is a report"
        # → subclass CombatInformation Report, not instance ?x CombatInformation)
        if pattern == "instance" and subject_class and object_class:
            if subject_class != object_class:
                return MappedClaim(
                    pattern="subclass",
                    subject_class=subject_class,
                    relation=None,
                    object_class=object_class,
                    quantifier=claim.quantifier,
                    negated=claim.negated,
                    source_span=claim.source_span,
                    confidence=claim.confidence,
                    subject_phrase=claim.subject,
                    predicate_phrase=claim.predicate,
                    object_phrase=claim.object,
                )

        # For instance/existential patterns, we need at least one class
        if pattern in ("instance", "existential"):
            # Prefer subject_class; fall back to object_class
            effective_class = subject_class or object_class
            if not effective_class:
                return None
            return MappedClaim(
                pattern=pattern,
                subject_class=effective_class,
                relation=None,
                object_class=None,
                quantifier=claim.quantifier,
                negated=claim.negated,
                source_span=claim.source_span,
                confidence=claim.confidence,
                subject_phrase=claim.subject,
                predicate_phrase=claim.predicate,
                object_phrase=claim.object,
            )

        if pattern == "subclass":
            if not subject_class or not object_class:
                return None

        if pattern == "relation":
            if not relation:
                return None
            if not subject_class and not object_class:
                return None

        return MappedClaim(
            pattern=pattern,
            subject_class=subject_class,
            relation=relation,
            object_class=object_class,
            quantifier=claim.quantifier,
            negated=claim.negated,
            source_span=claim.source_span,
            confidence=claim.confidence,
            subject_phrase=claim.subject,
            predicate_phrase=claim.predicate,
            object_phrase=claim.object,
        )


# ── KIF assembler ──────────────────────────────────────────────────────────


def assemble_kif(mapped: MappedClaim) -> AssembledFact | None:
    """Assemble a MappedClaim into CNL and KIF strings. Returns None on failure."""
    pattern = mapped.pattern
    subj = mapped.subject_class
    obj = mapped.object_class
    rel = mapped.relation
    neg = mapped.negated

    cnl: str | None = None
    kif: str | None = None
    terms: list[str] = []
    relation_name: str | None = None

    if pattern == "instance":
        if not subj:
            return None
        cnl = f"?x is-a {subj}"
        kif = f"(instance ?x {subj})"
        terms = [subj]
        if neg:
            cnl = f"not ?x is-a {subj}"
            kif = f"(not (instance ?x {subj}))"

    elif pattern == "existential":
        if not subj:
            return None
        cnl = f"some ?x is-a {subj}"
        kif = f"(exists (?x) (instance ?x {subj}))"
        terms = [subj]

    elif pattern == "subclass":
        if not subj or not obj:
            return None
        cnl = f"{subj} subclass-of {obj}"
        kif = f"(subclass {subj} {obj})"
        terms = [subj, obj]
        if neg:
            cnl = f"not {subj} subclass-of {obj}"
            kif = f"(not (subclass {subj} {obj}))"

    elif pattern == "relation":
        if not rel:
            return None
        relation_name = rel
        arg1 = subj or "?x"
        arg2 = obj or "?y"
        terms = [t for t in [rel, subj, obj] if t]

        if mapped.quantifier == "universal" and subj:
            # every ?x is-a Subj implies rel ?x Obj
            inner_rel = f"{rel} ?x {arg2}"
            inner_kif = f"({rel} ?x {arg2})"
            if neg:
                inner_kif = f"(not ({rel} ?x {arg2}))"
            cnl = f"every ?x is-a {subj} implies {inner_rel}"
            kif = f"(forall (?x) (=> (instance ?x {subj}) {inner_kif}))"
        else:
            cnl = f"{rel} {arg1} {arg2}"
            kif = f"({rel} {arg1} {arg2})"
            if neg:
                cnl = f"not {rel} {arg1} {arg2}"
                kif = f"(not ({rel} {arg1} {arg2}))"

    elif pattern == "conditional":
        # Conditional with relation
        if not rel or not subj:
            return None
        relation_name = rel
        arg2 = obj or "?y"
        terms = [t for t in [rel, subj, obj] if t]
        inner_kif_str = f"({rel} ?x {arg2})"
        if neg:
            inner_kif_str = f"(not ({rel} ?x {arg2}))"
        cnl = f"every ?x is-a {subj} implies {rel} ?x {arg2}"
        kif = f"(forall (?x) (=> (instance ?x {subj}) {inner_kif_str}))"

    else:
        return None

    if cnl is None or kif is None:
        return None

    return AssembledFact(
        cnl=cnl,
        kif=kif,
        pattern=pattern,
        relation=relation_name,
        terms=terms,
        source_span=mapped.source_span,
        confidence=mapped.confidence,
    )


