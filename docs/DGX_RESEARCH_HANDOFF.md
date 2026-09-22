# DGX research handoff

As of 2026-09-22, the current execution path is the rootless container setup
in `docs/CONTAINER_RESEARCH.md`. A 27-record training smoke test completed on
`spark-87dc`; the adapter reloaded successfully, but sample outputs did not
compile as CNL. The checkpoint and run manifest are available on the worker
and controller at `/home/save-water/nl2logic-state/outputs/smoke-20260922-03`.
The conda/tmux launcher below predates the container setup.

## While hardware is being prepared

The CPU pipeline defaults to source-only extraction. Review the proposed labels
in `data/benchmarks/fm2-0/chapter1-2023-review/REVIEW_BATCH_01.md`. Proposals remain
drafts until a person checks the source and records their decision. These examples
are development data, since they already informed the rules; reserve separate
documents for a blind evaluation before further tuning.

## First hardware check

On the DGX, in the project's Python environment, run:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); x=torch.ones(32, device='cuda'); print((x+x).sum().item())"
python -m pytest -q
python scripts/launch_training.py --list-presets
```

After the environment and CUDA check pass, the existing launcher can exercise a
single machine with the smoke preset:

```bash
python scripts/launch_training.py smoke --host local --repo-dir "$PWD"
```

The launcher expects the `nl2logic` conda environment and tmux. It replaces its
preset's tmux session, so run it when that session has no work to preserve.
This is an infrastructure smoke test, not a scientific comparison. Save the
training log, environment versions, repository revision and resulting checkpoint.
Check the actual training log before choosing the files for leakage checks.

## Before a research comparison

Freeze human-reviewed development and held-out splits, training inputs, ontology
and source-only policy. Then run (replace paths with those frozen artifacts):

```bash
python scripts/research_preflight.py --benchmark path/to/reviewed.jsonl --training-data path/to/actual-training.jsonl --output path/to/preflight.json
python scripts/compare_formalizers.py --benchmark path/to/reviewed.jsonl --training-data path/to/actual-training.jsonl --model-path path/to/checkpoint --methods rules raw constrained --background-policy source-only --output-dir path/to/new-comparison
```

Repeat `--training-data` for every training input. The comparison command remains
usable for diagnostics even when preflight fails; do not interpret those runs as
the planned research study. Raw and constrained arms must use the same checkpoint.
Preflight cannot certify semantic novelty, blind independence or reviewer quality.

## Remaining research work

- Human-reviewed source-faithful targets beyond existing ontology edges, with
  independently reserved documents and explicit annotation guidelines.
- Typed claim representation and relation argument checks; closure validation is
  currently available for gold preflight, not enforced across ingestion.
- Model runs and matched comparisons, followed by manual accepted-error analysis.
- A paper reporting measured precision/coverage, uncertainty and limitations.

Current engineering tests and seed diagnostics do not establish those results.
