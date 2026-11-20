"""Prompt construction for the 4 experimental conditions.

Every condition shares the same task block (rendered from the task spec, visible shapes
only) and differs only in the extra context injected after it:

1. baseline          — task description only.
2. format_card       — + the full format spec named by spec['format_ref'] (the bug-axis
                       documentation the task was built from).
3. unittest_feedback — + bit-exact/exact-code unit-test failures from previous iterations,
                       rendered as concise (expected vs got) lines.
4. expert_retrieval  — + a pointer to (and the relevant excerpt of) the expert kernel the
                       task's baseline is drawn from (Phase 1: vendored expert source).

The loop and conditions are identical across models so results stay comparable; see
STATUS.md RQ3.
"""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONDITIONS = ("baseline", "format_card", "unittest_feedback", "expert_retrieval")

_PREAMBLE = """\
You are implementing a quantized operator. Write a single Python function

    def kernel(**inputs) -> dict

that computes the operation below from scratch. numpy and torch are available; you may not
import the benchmark's own modules. Match the declared dtypes exactly. Where verification
is bit-exact, the operation has exactly one correct answer — rounding mode, packing order
and scale broadcast are part of the contract, not implementation details. Output only code.
"""

_TASK_BLOCK = """\
## Task: {name} (group: {group})

{description}

Inputs:  {inputs}
Outputs: {outputs}
Constraints: {constraints}

Shapes you must handle (hidden evaluation shapes are different and not shown):
{shapes}
"""

_EXPERT_EXCERPTS = {
    # Phase 1 replaces these pointers with retrieved excerpts of the vendored expert source.
    "marlin": "expert reference: Marlin (IST-DASLab/marlin), global->shared weight dequant path",
    "humming": "expert reference: vLLM Humming, WNA16/W8A8 GEMM strategies",
    "torchao": "expert reference: torchao quantization layouts",
    "bitsandbytes": "expert reference: bitsandbytes LLM.int8() outlier handling",
    "_scaled_mm": "expert reference: torch._scaled_mm scale semantics (cuBLASLt)",
    "null": "expert reference: none registered for this task",
}


def _shapes_block(spec):
    lines = []
    for kind in ("visible",):
        for s in spec["shapes"][kind]:
            lines.append("- " + ", ".join(f"{k}={v}" for k, v in s.items()))
    return "\n".join(lines)


def build_prompt(spec, condition, feedback=None):
    """Render the prompt for a task spec under one of the four conditions."""
    assert condition in CONDITIONS, condition
    parts = [_PREAMBLE, _TASK_BLOCK.format(
        name=spec["name"], group=spec["group"], description=spec["description"].strip(),
        inputs=spec["inputs"], outputs=spec["outputs"],
        constraints=spec.get("constraints", "-"), shapes=_shapes_block(spec))]
    mode = spec["verification"]["mode"]
    parts.append(f"Verification: {mode}."
                 + (f" Tolerance: {spec['verification']['tolerance']}" if mode != "bit_exact" else
                    " Outputs must match the reference byte for byte."))

    if condition == "format_card":
        ref = spec.get("format_ref")
        path = os.path.join(ROOT, ref) if ref else None
        if path and os.path.exists(path):
            with open(path) as f:
                parts.append("## Format spec\n\n" + f.read())
    elif condition == "unittest_feedback":
        if feedback:
            parts.append("## Failed checks from your previous attempt\n\n" + feedback)
    elif condition == "expert_retrieval":
        parts.append(_EXPERT_EXCERPTS.get(spec.get("baseline", {}).get("expert"), ""))
    return "\n\n".join(p for p in parts if p)


def render_feedback(failures, max_lines=6):
    """Condense verifier failures into the unit-test feedback block."""
    lines = [f"- {f}" for f in failures[:max_lines]]
    if len(failures) > max_lines:
        lines.append(f"- ... and {len(failures) - max_lines} more")
    return "\n".join(lines)
