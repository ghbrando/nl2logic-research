"""
generate_pairs.py — Training data generator for NL2Logic Stage 1.

Generates (NL, CNL, KIF) training pairs for T5 + LoRA fine-tuning.
Every generated CNL string is compiled through CNLCompiler to verify
correctness — invalid pairs are discarded.

The generator emits the following pattern buckets and production templates:
  instance     : ?x is-a {Class}
  subclass     : {Child} subclass-of {Parent}
  binary       : {rel} ?x ?y           (arity-2 relations only)
  nary         : {rel} [ ?x , ?y , ?z ] (arity-3 relations only)
  conditional  : every ?x is-a {Class} implies {rel} ?x {Type2}
  conditional  : if {rel} ?x {Type2} then ?x is-a {Class}
  existential  : some ?x is-a {Class}
  negation     : not {atomic_assertion}

Output format (JSONL, one record per line):
  {"nl": "...", "cnl": "...", "kif": "...", "pattern": "..."}

Usage:
    python src/data/generate_pairs.py [--limit N] [--output PATH]

Must be run from the repo root (CNLCompiler resolves vocab files relative
to the working directory).
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CLASSES_PATH = _REPO_ROOT / "data" / "training_pairs" / "sumo_classes.jsonl"
_RELATIONS_PATH = _REPO_ROOT / "data" / "training_pairs" / "sumo_relations.jsonl"
_OUTPUT_PATH = _REPO_ROOT / "data" / "training_pairs" / "nl_cnl_pairs.jsonl"
_DOCTRINE_KIF_PATH = _REPO_ROOT / "data" / "ontology" / "doctrine_domain.kif"

# CNL terminal constraints (mirroring cnl.lark exactly)
_CLASS_RE = re.compile(r"^[A-Z][a-zA-Z0-9_]*$")
_RELATION_RE = re.compile(r"^[a-z][a-zA-Z0-9]*$")

# Splits PascalCase / camelCase on word boundaries
_WORD_BOUNDARY_RE = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

# Strips &%TermName SUMO markup from doc strings
_SUMO_REF_RE = re.compile(r"&%(\S+)")

# Mutable discard counter shared across generators
_n_discarded = 0
_BALANCE_SEED = 42

# Context phrases for distractor augmentation (military-flavored)
_DISTRACTOR_CONTEXTS = [
    "in the operation",
    "in the force",
    "in the arsenal",
    "in the headquarters",
    "in the command post",
    "in planning",
    "at the headquarters",
    "for the operation",
    "in the theater",
    "in the field",
    "in the unit",
    "at the base",
]

# Relation-specific verb templates for natural NL phrasing.
# Key: relation name → (verb phrase with {t1}/{t2} slots, negated form)
_RELATION_VERB_TEMPLATES: dict[str, list[str]] = {
    "agent": [
        "A {t1} has a {t2}.",
        "A {t1_nat} has a {t2_nat}.",
        "A {t1_nat} has a {t2_nat} as its agent.",
    ],
    "patient": [
        "A {t1} can have a {t2} as its patient.",
        "A {t1_nat} can have a {t2_nat} as its patient.",
    ],
    "instrument": [
        "A {t1} uses a {t2} as an instrument.",
        "A {t1_nat} uses a {t2_nat} as an instrument.",
    ],
    "destination": [
        "A {t1} can have a {t2} as its destination.",
        "A {t1_nat} can have a {t2_nat} as its destination.",
    ],
    "origin": [
        "A {t1} can have a {t2} as its origin.",
        "A {t1_nat} can have a {t2_nat} as its origin.",
    ],
    "located": [
        "A {t1} can be located in a {t2}.",
        "A {t1_nat} can be located in a {t2_nat}.",
    ],
    "member": [
        "A {t1} is a member of a {t2}.",
        "A {t1_nat} is a member of a {t2_nat}.",
    ],
    "part": [
        "A {t1} is part of a {t2}.",
        "A {t1_nat} is part of a {t2_nat}.",
    ],
    "result": [
        "A {t1} results in a {t2}.",
        "A {t1_nat} results in a {t2_nat}.",
    ],
    "causes": [
        "A {t1} causes a {t2}.",
        "A {t1_nat} causes a {t2_nat}.",
    ],
    "resource": [
        "A {t1} requires a {t2} as a resource.",
        "A {t1_nat} requires a {t2_nat} as a resource.",
    ],
    "experiencer": [
        "A {t1} is experienced by a {t2}.",
        "A {t1_nat} is experienced by a {t2_nat}.",
    ],
}

# Negated relation verb templates
_NEGATED_RELATION_VERB_TEMPLATES: dict[str, list[str]] = {
    "agent": [
        "A {t1} does not have a {t2}.",
        "A {t1_nat} does not have a {t2_nat}.",
        "A {t1_nat} does not have a {t2_nat} as its agent.",
    ],
    "patient": [
        "A {t1_nat} does not have a {t2_nat} as its patient.",
    ],
    "instrument": [
        "A {t1_nat} does not use a {t2_nat} as an instrument.",
    ],
    "destination": [
        "A {t1_nat} does not have a {t2_nat} as its destination.",
    ],
    "origin": [
        "A {t1_nat} does not have a {t2_nat} as its origin.",
    ],
    "located": [
        "A {t1} is not located in a {t2}.",
        "A {t1_nat} is not located in a {t2_nat}.",
    ],
    "member": [
        "A {t1_nat} is not a member of a {t2_nat}.",
    ],
    "part": [
        "A {t1_nat} is not part of a {t2_nat}.",
    ],
}

# Subclass expansions for grounded-pair diversification.
# Maps SUMO signature types to military-relevant subclasses so that
# each relation produces multiple grounded CNL forms, not just the
# single signature-type pair.
_SUBCLASS_EXPANSIONS: dict[str, list[str]] = {
    "Process": [
        "MilitaryProcess", "Attack", "Battle", "Transportation",
        "Communication", "Order",
    ],
    "Object": [
        "Weapon", "MilitaryUnit", "Area", "Artifact", "Region",
    ],
    "Entity": [
        "Weapon", "MilitaryUnit", "Region", "Area",
        "AutonomousAgent", "MilitaryProcess",
    ],
    "Physical": [
        "MilitaryUnit", "Area", "Weapon", "Region",
    ],
    "AutonomousAgent": [
        "MilitaryUnit",
    ],
    "Agent": [
        "AutonomousAgent", "MilitaryUnit",
    ],
    "PositionalAttribute": [
        "Vertical",
    ],
    # Doctrine domain parents → doctrine child classes (doctrine_domain.kif)
    "MilitaryProcess": [
        "IntelligenceProcess", "WarfightingFunction", "IntelligenceWarfightingFunction",
        "ThreatCourseOfAction", "DoctrinalTask", "IntelligenceWarfightingFunctionTask",
        "InformationCollection",
    ],
    "MilitaryOrganization": [
        "IntelligenceEnterprise",
    ],
    "MilitaryPerson": [
        "IntelligenceProfessional",
    ],
    "ContentBearingObject": [
        "IntelligenceProduct",
    ],
    "Procedure": [
        "IntelligenceDiscipline", "HumanIntelligence", "SignalsIntelligence",
        "GeospatialIntelligence",
    ],
    "FactualText": [
        "CombatInformation",
    ],
    "Region": [
        "OperationalEnvironment",
    ],
    "Communication": [
        "IntelligenceDissemination",
    ],
    "MilitaryCommander": [
        "TacticalCommander",
    ],
}

_TARGETED_NL_BOOSTERS: dict[str, dict[str, list[str]]] = {
    "binary": {
        "agent MilitaryProcess AutonomousAgent": [
            "A military process has an autonomous agent in the agent role.",
            "A military process involves an autonomous agent as its agent.",
        ],
        "destination Transportation Region": [
            "A transportation process has a region as its destination.",
            "A transportation process can end in a region.",
        ],
        "origin Transportation Region": [
            "A transportation process has a region as its origin.",
            "A transportation process can start in a region.",
        ],
        "located MilitaryUnit Area": [
            "A military unit can be located in an area of operation.",
            "A military unit can be stationed in an operational area.",
        ],
    },
    "conditional": {
        "every ?x is-a MilitaryProcess implies agent ?x AutonomousAgent": [
            "Every military process has an autonomous agent in the agent role.",
            "Every military process involves an autonomous agent as its agent.",
        ],
        "every ?x is-a Transportation implies destination ?x Region": [
            "Every transportation process has a region as its destination endpoint.",
            "Every transportation process can end in a region.",
        ],
        "every ?x is-a Transportation implies origin ?x Region": [
            "Every transportation process has a region as its starting origin.",
            "Every transportation process can start in a region.",
        ],
    },
    "existential": {
        "some ?x is-a Plan": [
            "There exists a plan for an operation.",
            "At least one plan exists for an operation.",
        ],
    },
    "negation": {
        "not agent MilitaryProcess AutonomousAgent": [
            "A military process does not have an autonomous agent in the agent role.",
            "A military process lacks an autonomous agent as its agent.",
        ],
        "not located MilitaryUnit Area": [
            "A military unit is not located in an area of operation.",
            "A military unit is not stationed in an operational area.",
        ],
    },
}


def _expand_type(sig_type: str) -> list[str]:
    """Return *sig_type* plus its military-relevant subclass expansions."""
    return [sig_type] + _SUBCLASS_EXPANSIONS.get(sig_type, [])


# Conditional verb templates (every X has/uses Y)
_CONDITIONAL_VERB_TEMPLATES: dict[str, list[str]] = {
    "agent": [
        "Every {cls} has a {t2}.",
        "Every {cls_nat} has a {t2_nat}.",
        "Every {cls_nat} has a {t2_nat} as its agent.",
    ],
    "patient": [
        "Every {cls} can have a {t2} as its patient.",
        "Every {cls_nat} can have a {t2_nat} as its patient.",
    ],
    "instrument": [
        "Every {cls} uses a {t2} as an instrument.",
        "Every {cls_nat} uses a {t2_nat} as an instrument.",
    ],
    "destination": [
        "Every {cls} has a {t2} as its destination.",
        "Every {cls_nat} has a {t2_nat} as its destination.",
    ],
    "origin": [
        "Every {cls} has a {t2} as its origin.",
        "Every {cls_nat} has a {t2_nat} as its origin.",
    ],
    "located": [
        "Every {cls} is located in a {t2}.",
        "Every {cls_nat} is located in a {t2_nat}.",
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pascal_to_natural(term: str) -> str:
    """Convert PascalCase or camelCase to lowercase natural language.

    >>> pascal_to_natural("MilitaryProcess")
    'military process'
    >>> pascal_to_natural("AutonomousAgent")
    'autonomous agent'
    >>> pascal_to_natural("AAM")
    'aam'
    >>> pascal_to_natural("agent")
    'agent'
    """
    return _WORD_BOUNDARY_RE.sub(" ", term).lower()


def _article_for(word: str) -> str:
    """Return 'an' if *word* starts with a vowel sound, else 'a'."""
    return "an" if word and word[0].lower() in "aeiou" else "a"


def _clean_doc(doc: str) -> str:
    """Strip &%Ref markup, truncate to first sentence (≤300 chars).

    Returns empty string if the result is too short to be useful.
    """
    doc = _SUMO_REF_RE.sub(r"\1", doc).strip()
    # Take first sentence: find '. ' (period then space) anywhere
    m = re.search(r"\.\s", doc)
    if m:
        doc = doc[: m.start() + 1]
    doc = doc[:300].strip()
    return doc if len(doc) > 15 else ""


def _compile_safe(compiler, cnl: str) -> str | None:
    """Return KIF or None on any compiler error; increment discard counter."""
    global _n_discarded
    try:
        return compiler.compile(cnl)
    except Exception:
        _n_discarded += 1
        return None


def _pair(nl: str, cnl: str, kif: str, pattern: str) -> dict:
    return {"nl": nl, "cnl": cnl, "kif": kif, "pattern": pattern}


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def _extend_with_targeted_boosters(nls: list[str], *, pattern: str, cnl: str) -> list[str]:
    boosted = list(nls)
    boosted.extend(_TARGETED_NL_BOOSTERS.get(pattern, {}).get(cnl, []))
    return _dedupe_preserve_order(boosted)


def count_pairs_by_pattern(pairs: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for pair in pairs:
        pattern = pair["pattern"]
        counts[pattern] = counts.get(pattern, 0) + 1
    return counts


def select_output_pairs(
    pairs: list[dict],
    *,
    limit: int,
    balanced: bool,
    seed: int = _BALANCE_SEED,
) -> list[dict]:
    if limit <= 0:
        return list(pairs)

    if not balanced:
        return list(pairs[:limit])

    grouped: dict[str, list[dict]] = {}
    for pair in pairs:
        grouped.setdefault(pair["pattern"], []).append(pair)

    if not grouped:
        return []

    per_pattern_cap = limit // len(grouped)
    rng = random.Random(seed)
    selected: list[dict] = []

    for bucket in grouped.values():
        shuffled = list(bucket)
        rng.shuffle(shuffled)
        selected.extend(shuffled[:per_pattern_cap])

    return selected


# ---------------------------------------------------------------------------
# Vocabulary loading
# ---------------------------------------------------------------------------

def load_classes(path: Path) -> list[dict]:
    """Load class records whose terms match the CNL CLASS_TERM regex."""
    result = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if _CLASS_RE.match(rec["term"]):
                result.append(rec)
    return result


def load_relations(path: Path) -> list[dict]:
    """Load relation records whose terms match the CNL RELATION regex
    and have a known arity."""
    result = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if _RELATION_RE.match(rec["term"]) and rec.get("arity") is not None:
                result.append(rec)
    return result


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def gen_instance_pairs(
    classes: list[dict],
    compiler,
    *,
    include_doc_templates: bool = True,
) -> list[dict]:
    """?x is-a {Class}  →  (instance ?x {Class})"""
    pairs = []
    for rec in classes:
        cls = rec["term"]
        cls_nat = pascal_to_natural(cls)
        cnl = f"?x is-a {cls}"
        kif = _compile_safe(compiler, cnl)
        if kif is None:
            continue
        art = _article_for(cls_nat)
        nls = [
            # Original metalinguistic templates
            f"Something is an instance of {cls}.",
            f"There exists a {cls}.",
            f"An individual of type {cls} exists.",
            # Natural-language templates (gold-style)
            f"Something is {art} {cls_nat}.",
            f"There is {art} {cls_nat}.",
        ]
        # Distractor-augmented templates
        for ctx in _DISTRACTOR_CONTEXTS[:4]:
            nls.append(f"Something {ctx} is {art} {cls_nat}.")
            nls.append(f"Something {ctx} is an instance of {cls}.")
        if include_doc_templates and (doc := _clean_doc(rec.get("doc", ""))):
            nls.append(doc)
        for nl in nls:
            pairs.append(_pair(nl, cnl, kif, "instance"))
    return pairs


def gen_subclass_pairs(
    classes: list[dict],
    compiler,
    *,
    include_doc_templates: bool = True,
) -> list[dict]:
    """{Child} subclass-of {Parent}  →  (subclass {Child} {Parent})

    Pairs each class with the alphabetically adjacent class as its
    syntactic 'parent'.  The subclass relation is semantically arbitrary
    here — the model learns the NL→CNL form, not ontological truth.
    """
    pairs = []
    for i in range(len(classes) - 1):
        child = classes[i]["term"]
        parent = classes[i + 1]["term"]
        child_nat = pascal_to_natural(child)
        parent_nat = pascal_to_natural(parent)
        cnl = f"{child} subclass-of {parent}"
        kif = _compile_safe(compiler, cnl)
        if kif is None:
            continue
        art_child = _article_for(child_nat)
        art_parent = _article_for(parent_nat)
        nls = [
            # PascalCase templates (original)
            f"{child} is a type of {parent}.",
            f"Every {child} is a {parent}.",
            f"{child} is a subclass of {parent}.",
            # Natural-language templates (gold-style)
            f"A {child_nat} is a type of {parent_nat}.",
            f"{art_child.title()} {child_nat} is a type of {parent_nat}.",
            f"Every {child_nat} is {art_parent} {parent_nat}.",
            f"A {child_nat} is a subclass of {parent_nat}.",
        ]
        if include_doc_templates and (doc := _clean_doc(classes[i].get("doc", ""))):
            nls.append(doc)
        for nl in nls:
            pairs.append(_pair(nl, cnl, kif, "subclass"))
    return pairs


def gen_doctrine_subclass_pairs(
    doctrine_kif: Path,
    compiler,
) -> list[dict]:
    """{Child} subclass-of {Parent}  →  (subclass {Child} {Parent})

    Generates pairs from real ontological edges in doctrine_domain.kif.
    Every (subclass X Y) declaration becomes a set of NL paraphrases.
    Unlike gen_subclass_pairs, the child/parent relationship here is
    semantically correct, not alphabetically arbitrary.
    """
    from src.ontology.vocab import load_doctrine_subclass_pairs

    pairs = []
    edges = load_doctrine_subclass_pairs(doctrine_kif)
    for child, parent in edges.items():
        child_nat = pascal_to_natural(child)
        parent_nat = pascal_to_natural(parent)
        cnl = f"{child} subclass-of {parent}"
        kif = _compile_safe(compiler, cnl)
        if kif is None:
            continue
        art_child = _article_for(child_nat)
        art_parent = _article_for(parent_nat)
        nls = [
            f"{child} is a type of {parent}.",
            f"Every {child} is a {parent}.",
            f"{child} is a subclass of {parent}.",
            f"A {child_nat} is a type of {parent_nat}.",
            f"{art_child.title()} {child_nat} is a type of {parent_nat}.",
            f"Every {child_nat} is {art_parent} {parent_nat}.",
            f"A {child_nat} is a subclass of {parent_nat}.",
        ]
        for nl in nls:
            pairs.append(_pair(nl, cnl, kif, "subclass"))
    return pairs


def gen_binary_pairs(
    relations: list[dict],
    compiler,
    *,
    include_doc_templates: bool = True,
) -> list[dict]:
    """Emit both abstract-variable and grounded typed binary assertions."""
    pairs = []
    for rec in relations:
        if rec.get("arity") != 2:
            continue
        rel = rec["term"]
        sig = rec.get("signature", {})
        t1 = sig.get("1", "Entity")
        t2 = sig.get("2", "Entity")
        # Signature class terms may contain hyphens; fall back to Entity
        if not _CLASS_RE.match(t1):
            t1 = "Entity"
        if not _CLASS_RE.match(t2):
            t2 = "Entity"
        rel_nat = pascal_to_natural(rel)
        abstract_cnl = f"{rel} ?x ?y"
        abstract_kif = _compile_safe(compiler, abstract_cnl)
        if abstract_kif is not None:
            abstract_nls = [
                f"The {rel} relation holds between something and something else.",
                f"{rel} relates one thing to another.",
                f"Something stands in the {rel} relation to something else.",
            ]
            for nl in abstract_nls:
                pairs.append(_pair(nl, abstract_cnl, abstract_kif, "binary"))

        # Generate grounded pairs for signature types AND expanded subclasses
        seen_grounded: set[str] = set()
        for g1 in _expand_type(t1):
            for g2 in _expand_type(t2):
                grounded_cnl = f"{rel} {g1} {g2}"
                if grounded_cnl in seen_grounded:
                    continue
                seen_grounded.add(grounded_cnl)
                grounded_kif = _compile_safe(compiler, grounded_cnl)
                if grounded_kif is None:
                    continue

                g1_nat = pascal_to_natural(g1)
                g2_nat = pascal_to_natural(g2)
                art1 = _article_for(g1_nat)
                art2 = _article_for(g2_nat)
                grounded_nls = [
                    # Metalinguistic templates (original)
                    f"The {rel} relation holds between a {g1} and a {g2}.",
                    f"{rel} relates a {g1} to a {g2}.",
                    f"A {g1} stands in the {rel} relation to a {g2}.",
                    # Natural-language templates (gold-style)
                    f"The {rel} relation holds between {art1} {g1_nat} and {art2} {g2_nat}.",
                    f"{art1.title()} {g1_nat} stands in the {rel} relation to {art2} {g2_nat}.",
                    f"{art1.title()} {g1_nat} has {art2} {g2_nat} as its {rel_nat}.",
                ]
                # Relation-specific verb templates
                if rel in _RELATION_VERB_TEMPLATES:
                    fmt_vars = {"t1": g1, "t2": g2, "t1_nat": g1_nat, "t2_nat": g2_nat}
                    for tmpl in _RELATION_VERB_TEMPLATES[rel]:
                        grounded_nls.append(tmpl.format(**fmt_vars))
                # Doc template only for the primary signature pair
                if g1 == t1 and g2 == t2:
                    if include_doc_templates and (doc := _clean_doc(rec.get("doc", ""))):
                        grounded_nls.append(doc)
                grounded_nls = _extend_with_targeted_boosters(
                    grounded_nls,
                    pattern="binary",
                    cnl=grounded_cnl,
                )
                for nl in grounded_nls:
                    pairs.append(_pair(nl, grounded_cnl, grounded_kif, "binary"))
    return pairs


def gen_nary_pairs(
    relations: list[dict],
    compiler,
    *,
    include_doc_templates: bool = True,
) -> list[dict]:
    """Emit both abstract-variable and grounded typed n-ary assertions."""
    pairs = []
    for rec in relations:
        if rec.get("arity") != 3:
            continue
        rel = rec["term"]
        sig = rec.get("signature", {})
        t1 = sig.get("1", "Entity")
        t2 = sig.get("2", "Entity")
        t3 = sig.get("3", "Entity")
        if not _CLASS_RE.match(t1):
            t1 = "Entity"
        if not _CLASS_RE.match(t2):
            t2 = "Entity"
        if not _CLASS_RE.match(t3):
            t3 = "Entity"
        abstract_cnl = f"{rel} [ ?x , ?y , ?z ]"
        abstract_kif = _compile_safe(compiler, abstract_cnl)
        if abstract_kif is not None:
            abstract_nls = [
                f"The {rel} relation holds among three entities.",
                f"{rel} connects three entities.",
                f"Three entities stand in the {rel} relation.",
            ]
            for nl in abstract_nls:
                pairs.append(_pair(nl, abstract_cnl, abstract_kif, "nary"))

        # Generate grounded pairs for signature types AND expanded subclasses
        seen_grounded: set[str] = set()
        for g1 in _expand_type(t1):
            for g2 in _expand_type(t2):
                for g3 in _expand_type(t3):
                    grounded_cnl = f"{rel} [ {g1} , {g2} , {g3} ]"
                    if grounded_cnl in seen_grounded:
                        continue
                    seen_grounded.add(grounded_cnl)
                    grounded_kif = _compile_safe(compiler, grounded_cnl)
                    if grounded_kif is None:
                        continue

                    g1_nat = pascal_to_natural(g1)
                    g2_nat = pascal_to_natural(g2)
                    g3_nat = pascal_to_natural(g3)
                    art1 = _article_for(g1_nat)
                    art2 = _article_for(g2_nat)
                    art3 = _article_for(g3_nat)
                    grounded_nls = [
                        # Metalinguistic (original)
                        f"The {rel} relation holds among a {g1}, a {g2}, and a {g3}.",
                        f"{rel} connects a {g1}, a {g2}, and a {g3}.",
                        f"A {g1}, a {g2}, and a {g3} stand in the {rel} relation.",
                        # Natural-language (gold-style)
                        f"The {rel} relation holds among {art1} {g1_nat}, {art2} {g2_nat}, and {art3} {g3_nat}.",
                        f"{rel} connects {art1} {g1_nat}, {art2} {g2_nat}, and {art3} {g3_nat}.",
                    ]
                    # Doc template only for the primary signature pair
                    if g1 == t1 and g2 == t2 and g3 == t3:
                        if include_doc_templates and (doc := _clean_doc(rec.get("doc", ""))):
                            grounded_nls.append(doc)
                    for nl in grounded_nls:
                        pairs.append(_pair(nl, grounded_cnl, grounded_kif, "nary"))
    return pairs


def gen_conditional_every_pairs(relations: list[dict], compiler) -> list[dict]:
    """every ?x is-a {Class} implies {rel} ?x {Type2}"""
    pairs = []
    for rec in relations:
        if rec.get("arity") != 2:
            continue
        rel = rec["term"]
        sig = rec.get("signature", {})
        cls = sig.get("1", "Entity")
        t2 = sig.get("2", "Entity")
        if not _CLASS_RE.match(cls):
            cls = "Entity"
        if not _CLASS_RE.match(t2):
            t2 = "Entity"
        rel_nat = pascal_to_natural(rel)

        # Generate for signature types AND expanded subclasses
        seen: set[str] = set()
        for gc in _expand_type(cls):
            for g2 in _expand_type(t2):
                cnl = f"every ?x is-a {gc} implies {rel} ?x {g2}"
                if cnl in seen:
                    continue
                seen.add(cnl)
                kif = _compile_safe(compiler, cnl)
                if kif is None:
                    continue
                gc_nat = pascal_to_natural(gc)
                g2_nat = pascal_to_natural(g2)
                art2 = _article_for(g2_nat)
                nls = [
                    # Metalinguistic (original)
                    f"Every {gc} is in the {rel} relation with a {g2}.",
                    f"For every instance of {gc}, the {rel} relation holds with a {g2}.",
                    f"All instances of {gc} are related to a {g2} via {rel}.",
                    # Natural-language (gold-style)
                    f"Every {gc_nat} is in the {rel} relation with {art2} {g2_nat}.",
                    f"Every {gc_nat} has {art2} {g2_nat} as its {rel_nat}.",
                ]
                # Relation-specific verb templates
                if rel in _CONDITIONAL_VERB_TEMPLATES:
                    fmt_vars = {"cls": gc, "t2": g2, "cls_nat": gc_nat, "t2_nat": g2_nat}
                    for tmpl in _CONDITIONAL_VERB_TEMPLATES[rel]:
                        nls.append(tmpl.format(**fmt_vars))
                nls = _extend_with_targeted_boosters(
                    nls,
                    pattern="conditional",
                    cnl=cnl,
                )
                for nl in nls:
                    pairs.append(_pair(nl, cnl, kif, "conditional"))
    return pairs


def gen_conditional_if_pairs(relations: list[dict], compiler) -> list[dict]:
    """if {rel} ?x {Type2} then ?x is-a {Class}"""
    pairs = []
    for rec in relations:
        if rec.get("arity") != 2:
            continue
        rel = rec["term"]
        sig = rec.get("signature", {})
        cls = sig.get("1", "Entity")
        t2 = sig.get("2", "Entity")
        if not _CLASS_RE.match(cls):
            cls = "Entity"
        if not _CLASS_RE.match(t2):
            t2 = "Entity"
        # Generate for signature types AND expanded subclasses
        seen: set[str] = set()
        for gc in _expand_type(cls):
            for g2 in _expand_type(t2):
                cnl = f"if {rel} ?x {g2} then ?x is-a {gc}"
                if cnl in seen:
                    continue
                seen.add(cnl)
                kif = _compile_safe(compiler, cnl)
                if kif is None:
                    continue
                gc_nat = pascal_to_natural(gc)
                g2_nat = pascal_to_natural(g2)
                art_cls = _article_for(gc_nat)
                art_t2 = _article_for(g2_nat)
                nls = [
                    # Metalinguistic (original)
                    f"If something stands in the {rel} relation to a {g2}, then it is a {gc}.",
                    f"If the {rel} relation holds from something to a {g2}, then the first argument is a {gc}.",
                    f"Anything that {rel}s a {g2} is a {gc}.",
                    # Natural-language (gold-style)
                    f"If something stands in the {rel} relation to {art_t2} {g2_nat}, then it is {art_cls} {gc_nat}.",
                    f"Anything that {rel}s {art_t2} {g2_nat} is {art_cls} {gc_nat}.",
                ]
                for nl in nls:
                    pairs.append(_pair(nl, cnl, kif, "conditional"))
    return pairs


def gen_existential_pairs(classes: list[dict], compiler) -> list[dict]:
    """some ?x is-a {Class}  →  (exists (?x) (instance ?x {Class}))"""
    pairs = []
    for rec in classes:
        cls = rec["term"]
        cls_nat = pascal_to_natural(cls)
        art = _article_for(cls_nat)
        cnl = f"some ?x is-a {cls}"
        kif = _compile_safe(compiler, cnl)
        if kif is None:
            continue
        nls = [
            # Original templates
            f"There exists an instance of {cls}.",
            f"Some entity is a {cls}.",
            f"At least one {cls} exists.",
            # Natural-language templates (gold-style)
            f"There exists {art} {cls_nat}.",
            f"At least one {cls_nat} exists.",
        ]
        # Distractor-augmented templates
        for ctx in _DISTRACTOR_CONTEXTS[:4]:
            nls.append(f"There exists an instance of {cls} {ctx}.")
            nls.append(f"There exists {art} {cls_nat} {ctx}.")
        nls = _extend_with_targeted_boosters(
            nls,
            pattern="existential",
            cnl=cnl,
        )
        for nl in nls:
            pairs.append(_pair(nl, cnl, kif, "existential"))
    return pairs


def gen_negation_pairs(classes: list[dict], relations: list[dict], compiler) -> list[dict]:
    """not {atomic_assertion}  →  (not {compiled_atomic_assertion})"""
    pairs = []

    for rec in classes:
        cls = rec["term"]
        cls_nat = pascal_to_natural(cls)
        art = _article_for(cls_nat)
        cnl = f"not ?x is-a {cls}"
        kif = _compile_safe(compiler, cnl)
        if kif is None:
            continue
        nls = [
            # Original templates
            f"Something is not a {cls}.",
            f"It is not the case that something is an instance of {cls}.",
            f"The entity is not a {cls}.",
            # Natural-language templates (gold-style)
            f"Something is not {art} {cls_nat}.",
        ]
        # Distractor-augmented templates
        for ctx in _DISTRACTOR_CONTEXTS[:2]:
            nls.append(f"Something {ctx} is not {art} {cls_nat}.")
            nls.append(f"Something {ctx} is not a {cls}.")
        for nl in nls:
            pairs.append(_pair(nl, cnl, kif, "negation"))

    for rec in relations:
        rel = rec["term"]
        sig = rec.get("signature", {})

        if rec.get("arity") == 2:
            t1 = sig.get("1", "Entity")
            t2 = sig.get("2", "Entity")
            if not _CLASS_RE.match(t1):
                t1 = "Entity"
            if not _CLASS_RE.match(t2):
                t2 = "Entity"
            abstract_cnl = f"not {rel} ?x ?y"
            abstract_kif = _compile_safe(compiler, abstract_cnl)
            if abstract_kif is not None:
                abstract_nls = [
                    f"The {rel} relation does not hold between something and something else.",
                    f"Something is not in the {rel} relation with something else.",
                    f"It is not the case that {rel} relates one thing to another.",
                ]
                for nl in abstract_nls:
                    pairs.append(_pair(nl, abstract_cnl, abstract_kif, "negation"))

            # Generate grounded pairs for signature types AND expanded subclasses
            seen_bin: set[str] = set()
            for g1 in _expand_type(t1):
                for g2 in _expand_type(t2):
                    grounded_cnl = f"not {rel} {g1} {g2}"
                    if grounded_cnl in seen_bin:
                        continue
                    seen_bin.add(grounded_cnl)
                    grounded_kif = _compile_safe(compiler, grounded_cnl)
                    if grounded_kif is None:
                        continue
                    g1_nat = pascal_to_natural(g1)
                    g2_nat = pascal_to_natural(g2)
                    art1 = _article_for(g1_nat)
                    art2 = _article_for(g2_nat)
                    grounded_nls = [
                        # Metalinguistic (original)
                        f"The {rel} relation does not hold between a {g1} and a {g2}.",
                        f"A {g1} is not in the {rel} relation with a {g2}.",
                        f"It is not the case that {rel} relates a {g1} to a {g2}.",
                        # Natural-language (gold-style)
                        f"The {rel} relation does not hold between {art1} {g1_nat} and {art2} {g2_nat}.",
                        f"{art1.title()} {g1_nat} is not in the {rel} relation with {art2} {g2_nat}.",
                    ]
                    # Relation-specific negated verb templates
                    if rel in _NEGATED_RELATION_VERB_TEMPLATES:
                        fmt_vars = {"t1": g1, "t2": g2, "t1_nat": g1_nat, "t2_nat": g2_nat}
                        for tmpl in _NEGATED_RELATION_VERB_TEMPLATES[rel]:
                            grounded_nls.append(tmpl.format(**fmt_vars))
                    grounded_nls = _extend_with_targeted_boosters(
                        grounded_nls,
                        pattern="negation",
                        cnl=grounded_cnl,
                    )
                    for nl in grounded_nls:
                        pairs.append(_pair(nl, grounded_cnl, grounded_kif, "negation"))

        if rec.get("arity") == 3:
            t1 = sig.get("1", "Entity")
            t2 = sig.get("2", "Entity")
            t3 = sig.get("3", "Entity")
            if not _CLASS_RE.match(t1):
                t1 = "Entity"
            if not _CLASS_RE.match(t2):
                t2 = "Entity"
            if not _CLASS_RE.match(t3):
                t3 = "Entity"
            abstract_cnl = f"not {rel} [ ?x , ?y , ?z ]"
            abstract_kif = _compile_safe(compiler, abstract_cnl)
            if abstract_kif is not None:
                abstract_nls = [
                    f"The {rel} relation does not hold among three entities.",
                    f"It is not the case that {rel} connects three entities.",
                    f"Three entities do not stand in the {rel} relation.",
                ]
                for nl in abstract_nls:
                    pairs.append(_pair(nl, abstract_cnl, abstract_kif, "negation"))

            # Generate grounded pairs for signature types AND expanded subclasses
            seen_nary: set[str] = set()
            for g1 in _expand_type(t1):
                for g2 in _expand_type(t2):
                    for g3 in _expand_type(t3):
                        grounded_cnl = f"not {rel} [ {g1} , {g2} , {g3} ]"
                        if grounded_cnl in seen_nary:
                            continue
                        seen_nary.add(grounded_cnl)
                        grounded_kif = _compile_safe(compiler, grounded_cnl)
                        if grounded_kif is None:
                            continue
                        g1_nat = pascal_to_natural(g1)
                        g2_nat = pascal_to_natural(g2)
                        g3_nat = pascal_to_natural(g3)
                        art1 = _article_for(g1_nat)
                        art2 = _article_for(g2_nat)
                        art3 = _article_for(g3_nat)
                        grounded_nls = [
                            # Metalinguistic (original)
                            f"The {rel} relation does not hold among a {g1}, a {g2}, and a {g3}.",
                            f"It is not the case that {rel} connects a {g1}, a {g2}, and a {g3}.",
                            f"A {g1}, a {g2}, and a {g3} do not stand in the {rel} relation.",
                            # Natural-language (gold-style)
                            f"The {rel} relation does not hold among {art1} {g1_nat}, {art2} {g2_nat}, and {art3} {g3_nat}.",
                        ]
                        for nl in grounded_nls:
                            pairs.append(_pair(nl, grounded_cnl, grounded_kif, "negation"))

    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate (NL, CNL, KIF) training pairs for NL2Logic Stage 1."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        metavar="N",
        help="Cap total output pairs (0 = no limit; useful for quick test runs)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(_OUTPUT_PATH),
        metavar="PATH",
        help=f"Output JSONL path (default: {_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="Sample up to limit // n_patterns pairs per pattern instead of taking the first N rows.",
    )
    parser.add_argument(
        "--include-doc-templates",
        action="store_true",
        help="Include raw SUMO doc-string NL templates in addition to the curated synthetic paraphrases.",
    )
    args = parser.parse_args()

    # Guard: compiler resolves vocab files relative to CWD
    if not (_REPO_ROOT / "data" / "training_pairs" / "sumo_classes.jsonl").exists():
        print("ERROR: Must be run from the repo root.", file=sys.stderr)
        sys.exit(1)

    # Ensure repo root is on sys.path for `src.*` imports
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

    from src.compiler.compiler import CNLCompiler
    compiler = CNLCompiler()

    print("Loading vocabulary...")
    classes = load_classes(_CLASSES_PATH)
    relations = load_relations(_RELATIONS_PATH)
    print(f"  Valid classes:   {len(classes):>6}")
    print(f"  Valid relations: {len(relations):>6}")

    generators = [
        ("instance",    gen_instance_pairs(classes, compiler, include_doc_templates=args.include_doc_templates)),
        ("subclass",    gen_subclass_pairs(classes, compiler, include_doc_templates=args.include_doc_templates)),
        ("subclass",    gen_doctrine_subclass_pairs(_DOCTRINE_KIF_PATH, compiler)),
        ("binary",      gen_binary_pairs(relations, compiler, include_doc_templates=args.include_doc_templates)),
        ("nary",        gen_nary_pairs(relations, compiler, include_doc_templates=args.include_doc_templates)),
        ("conditional", gen_conditional_every_pairs(relations, compiler)),
        ("conditional", gen_conditional_if_pairs(relations, compiler)),
        ("existential", gen_existential_pairs(classes, compiler)),
        ("negation",    gen_negation_pairs(classes, relations, compiler)),
    ]

    print("\nGenerating pairs...")
    all_pairs: list[dict] = []
    generated_counts: dict[str, int] = {}
    limit_hit = False

    for label, batch in generators:
        generated_counts[label] = generated_counts.get(label, 0) + len(batch)
        all_pairs.extend(batch)
        print(f"  {label:15s}: {len(batch):>7,} pairs  (total so far: {len(all_pairs):>8,})")
        if args.limit and not args.balanced and len(all_pairs) >= args.limit:
            all_pairs = all_pairs[: args.limit]
            limit_hit = True
            break

    if args.limit and not args.balanced and not limit_hit:
        all_pairs = all_pairs[: args.limit]

    output_pairs = select_output_pairs(all_pairs, limit=args.limit, balanced=args.balanced)
    written_counts = count_pairs_by_pattern(output_pairs)
    balanced_cap = (
        args.limit // len(generated_counts)
        if args.balanced and args.limit and generated_counts
        else None
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in output_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\n{'=' * 48}")
    print(f"  Training Pair Summary")
    print(f"{'=' * 48}")
    print(f"  Total pairs written:  {len(output_pairs):>10,}")
    print(f"  Pairs discarded:      {_n_discarded:>10,}  (compile errors)")
    if balanced_cap is not None:
        print(f"  Balanced cap/pattern: {balanced_cap:>10,}")
    print(f"\n  By pattern written:")
    for pattern in generated_counts:
        print(f"    {pattern:15s}: {written_counts.get(pattern, 0):>10,}")
    print(f"\n  By pattern generated before limit:")
    for pattern, count in generated_counts.items():
        print(f"    {pattern:15s}: {count:>10,}")
    print(f"{'=' * 48}")
    print(f"\n  Output: {output_path}")


if __name__ == "__main__":
    main()
