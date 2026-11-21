# harness/

- `validate_tasks.py` — task-spec validator: schema completeness, gen/ref key agreement with
  the spec, output sizes/dtypes, byte-level determinism, over every visible + hidden shape
  at seeds 0-2. Regenerates `tasks/manifest.json`. (`make test` runs it.)
- `run.py` — run driver: executes the agent loop per task × condition × model, evaluates the
  final candidate on the hidden shapes it never saw, and appends one JSONL record per
  attempt series to `runs/`. Phase 0 drives the fake models; Phase 1 swaps in real API
  clients and GPU execution.
