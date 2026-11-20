"""The generation loop: a fixed 6-iteration cycle, identical for every model.

    generate -> run against visible shapes -> verifier feedback -> revise

Each iteration the model sees the condition prompt plus (from iteration 2 on) the exact
verifier failures of its previous attempt. A candidate passes when:

- it passes the verifier on all visible shapes (seeds 0-2), AND
- it passes on the hidden shapes (seeds 0-2) during final evaluation, AND
- it contains no forbidden imports (the candidate may not import the oracle stack) and
  no output aliases an input.

The iteration budget and feedback format never vary across models or conditions, so
iteration counts are comparable (STATUS.md RQ1/RQ3). Phase 0 executes candidates in-process
under the pinned environment; Phase 1 moves execution into a per-attempt subprocess on the
GPU instance.
"""

import numpy as np

from prompts.build import build_prompt, render_feedback
from verifier.anticheat import DEFAULT_FORBIDDEN, scan_source
from verifier.check import check_outputs

MAX_ITERS = 6


class Attempt:
    def __init__(self, iteration, source, ok, failures):
        self.iteration = iteration
        self.source = source
        self.ok = ok
        self.failures = failures


def run_task(model_fn, spec, impl, condition, forbidden=DEFAULT_FORBIDDEN, max_iters=MAX_ITERS):
    """Drive one task to completion (or budget exhaustion). Returns (attempts, final)."""
    history = []
    feedback = None
    for it in range(1, max_iters + 1):
        prompt = build_prompt(spec, condition, feedback=feedback)
        src = model_fn(prompt, history)
        attempt = _evaluate_source(src, spec, impl, it, forbidden)
        history.append(attempt)
        if attempt.ok:
            return history, attempt
        feedback = render_feedback(attempt.failures)
    return history, history[-1]


def _evaluate_source(src, spec, impl, iteration, forbidden):
    violations = scan_source(src, forbidden)
    if violations:
        return Attempt(iteration, src, False,
                       [f"forbidden import/call: {v}" for v in violations])
    try:
        kernel = _compile_kernel(src)
    except SyntaxError as e:
        return Attempt(iteration, src, False, [f"syntax error: {e}"])
    failures = []
    for shape in spec["shapes"]["visible"]:
        for seed in (0, 1, 2):
            inputs = impl.gen_inputs(shape, seed)
            try:
                cand = kernel(**inputs)
            except Exception as e:  # noqa: BLE001
                failures.append(f"{shape} seed{seed}: {type(e).__name__}: {e}")
                continue
            ref = impl.ref(**inputs)
            ok, f = check_outputs(spec, impl, inputs, ref, cand)
            failures.extend(f"{shape} seed{seed}: {x}" for x in f)
    return Attempt(iteration, src, not failures, failures)


def _compile_kernel(src):
    ns = {"np": np, "numpy": np}
    try:
        import torch
        ns["torch"] = torch
    except ImportError:
        pass
    exec(compile(src, "<kernel>", "exec"), ns)  # noqa: S102 — Phase 0 in-process; Phase 1 sandboxes
    return ns["kernel"]
