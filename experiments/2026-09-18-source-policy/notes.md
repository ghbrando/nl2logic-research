# Source-policy diagnostic

Run the commands in config.json in order from the repository root with the same
Python environment. Each argv array follows the Python executable. Preflight is
expected to exit 2 because the seed positives restate existing ontology edges.
Comparison output directories must not already exist; use fresh paths for repeats.

The per-policy summary.json files contain the measured results; manifests contain
input/code hashes and environments. results.json is the data-readiness report,
not a model-training result. This diagnostic uses the available doctrine training
file for overlap checking, not a claim about any future checkpoint's training set.

These seed cases influenced system development. Their scores cannot support
held-out accuracy or publication claims. Lower acceptance after removing implied
parents is an intended consequence of enforcing source evidence.

Observed: all 15 seed records were scored. The legacy rules baseline accepted
3 (all direct ontology restatements), reviewed 1 and abstained on 11. Source-only
rules abstained on all 15. Neither policy accepted a novel target. Preflight found
4 reviewed positives, all direct restatements, and no exact-text overlap with the
supplied training file. It correctly reported blocked research readiness.
