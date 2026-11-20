# prompts/

Prompt construction (`build.py`) for the 4 experimental conditions — one doc per condition:

1. `baseline.md` — task description only (control)
2. `format_card.md` — + the task's format spec from `formats/`
3. `unittest_feedback.md` — + verifier failures from previous iterations
4. `expert_retrieval.md` — + expert kernel reference/excerpt

The shared preamble fixes the contract (no benchmark imports, exact dtypes, rounding modes
are part of the spec). Everything renders from `spec.yaml`, so prompts cannot drift from
tasks.
