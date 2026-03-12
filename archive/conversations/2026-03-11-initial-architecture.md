## Your One Paragraph Summary

The central insight of this pipeline is that direct
NL→KIF translation fails structurally because LLMs
cannot simultaneously handle semantic understanding
and strict syntax generation. Introducing a CNL
intermediate representation resolves this by acting
as a structured middleman — the model is trained and
constrained to output only valid CNL, allowing it to
focus entirely on semantics rather than syntactics.
The FSM guarantees two things: syntactic precision
100% of the time, and elimination of SUMO term
hallucinations via vocabulary-level logit masking.
Together these properties make the system fully
deterministic given valid CNL input, which matters
critically in high-assurance domains like military,
legal, and medical applications where hallucinated
logic is not an acceptable failure mode.
