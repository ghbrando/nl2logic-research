"""Rules and matched raw/constrained generation without access to labels."""
from __future__ import annotations

from src.fsm.cnl_fsm import CNLSampler, _DEFAULT_CONSTRAINED_NUM_BEAMS, _DEFAULT_CONSTRAINED_LENGTH_PENALTY
from src.training.train import format_prompt


class RulesPredictor:
    """Use existing prompt-router templates only when every slot is unique.

    This deliberately excludes the decoder's combinatorial fallback grammar.
    Doctrine parent mappings are shared background knowledge, not learned facts.
    """
    def __init__(self):
        self.router = CNLSampler(None, None)

    def __call__(self, nl: str) -> str | None:
        prompt = format_prompt(nl)
        enumerated = self.router.infer_enumerated_subclass_cnl(prompt)
        if enumerated is not None:
            return enumerated
        plan = self.router._infer_prompt_plan(prompt)
        if plan is None:
            return None
        slots = [plan.first_terms, plan.second_terms]
        if plan.template != "subclass":
            slots.append(plan.relation_terms)
        if plan.template in {"nary", "negation_nary"}:
            slots.append(plan.third_terms)
        if any(len(slot) != 1 for slot in slots):
            return None
        a, b = plan.first_terms[0], plan.second_terms[0]
        if plan.template == "subclass":
            return f"{a} subclass-of {b}"
        relation = plan.relation_terms[0]
        if plan.template == "conditional_every":
            return f"every ?x is-a {a} implies {relation} ?x {b}"
        if plan.template in {"binary", "negation_binary"}:
            body = f"{relation} {a} {b}"
        elif plan.template in {"nary", "negation_nary"}:
            body = f"{relation} [ {a} , {b} , {plan.third_terms[0]} ]"
        else:
            return None
        return f"not {body}" if plan.template.startswith("negation_") else body


class Seq2SeqPredictor:
    def __init__(self, model, tokenizer, *, constrained: bool, max_tokens: int = 200):
        self.model, self.tokenizer = model, tokenizer
        self.constrained, self.max_tokens = constrained, max_tokens
        self.sampler = CNLSampler(model, tokenizer)

    def __call__(self, nl: str) -> str | None:
        prompt = format_prompt(nl)
        if self.constrained:
            # No raw fallback; errors are reported by the harness.
            return self.sampler.sample(prompt, max_tokens=self.max_tokens)
        import torch
        # Match tokenization, beam search, length penalty, and token budget;
        # the raw arm omits only the prefix constraint function.
        encoded = self.sampler._tokenize_prompt(prompt)
        with torch.no_grad():
            output = self.model.generate(
                **encoded, max_new_tokens=self.max_tokens,
                num_beams=_DEFAULT_CONSTRAINED_NUM_BEAMS,
                length_penalty=_DEFAULT_CONSTRAINED_LENGTH_PENALTY,
                early_stopping=True, renormalize_logits=True,
            )
        decoded = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        return decoded[0].strip() if decoded else None
