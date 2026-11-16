"""Correctness checking for generated kernels.

Two verification modes, chosen per task in spec.yaml:

- bit_exact: every output must be byte-identical to the reference. Used where the operation
  has an exact definition (pack/unpack, integer quantization with a specified rounding mode,
  fp8 casts compared as uint8 views).
- derived_tolerance: integer outputs must match exactly (or within int_shift codes for
  float-math tasks where reduction order can flip a code at a halfway point); float outputs
  must stay within a tolerance derived from the quantization format's error bound — either
  sum_abs (coef * format error scale + atol) or elementwise (rtol*|ref| + atol). The error
  scale comes from the task impl's tol_scale(), which encodes the format bound (e.g. half a
  code step per weight weighted by |x|, or the fp8 mantissa error).

Never torch.allclose defaults. See formats/rounding_and_scale_conventions.md for the bug
axes each bound is designed to catch.
"""

import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_task(group, name):
    """spec dict + impl module for tasks/<group>/<name>/."""
    import importlib

    import yaml

    tdir = os.path.join(ROOT, "tasks", group, name)
    with open(os.path.join(tdir, "spec.yaml")) as f:
        spec = yaml.safe_load(f)
    impl = importlib.import_module(f"tasks.{group}.{name}.impl")
    return spec, impl


def check_outputs(spec, impl, inputs, ref_out, cand_out):
    """Compare a candidate's outputs against the reference. Returns (ok, failures list)."""
    tol = spec.get("verification", {}).get("tolerance", {})
    kind = tol.get("kind", "sum_abs")
    coef = float(tol.get("coef", 1.0))
    rtol = float(tol.get("rtol", 0.0))
    atol = float(tol.get("atol", 0.0))
    int_shift = int(tol.get("int_shift", 0))
    scale = {}
    if kind == "sum_abs" and hasattr(impl, "tol_scale"):
        scale = impl.tol_scale(**inputs)

    failures = []
    for key, ref in ref_out.items():
        cand = cand_out.get(key)
        if cand is None:
            failures.append(f"{key}: missing from candidate outputs")
            continue
        ref_a = np.ascontiguousarray(ref)
        cand_a = np.ascontiguousarray(cand)
        if cand_a.shape != ref_a.shape:
            failures.append(f"{key}: shape {cand_a.shape} != {ref_a.shape}")
            continue
        if np.issubdtype(ref_a.dtype, np.integer) or ref_a.dtype == np.uint8:
            diff = np.abs(cand_a.astype(np.int64) - ref_a.astype(np.int64))
            bad = diff > int_shift
            if bad.any():
                i = int(np.flatnonzero(bad.reshape(-1))[0])
                failures.append(
                    f"{key}: integer mismatch beyond ±{int_shift} at flat {i}: "
                    f"ref {ref_a.reshape(-1)[i]} cand {cand_a.reshape(-1)[i]}")
            continue
        # float path
        ref_f = ref_a.astype(np.float64)
        cand_f = cand_a.astype(np.float64)
        err = np.abs(cand_f - ref_f)
        if kind == "sum_abs":
            bound = coef * np.broadcast_to(np.asarray(scale.get(key, 0.0), np.float64),
                                           ref_f.shape) + atol
        else:
            bound = rtol * np.abs(ref_f) + atol
        nan_ref = np.isnan(ref_f)
        nan_ok = np.isnan(cand_f) | (np.abs(cand_f) > 1e30)  # a NaN or inf where ref is NaN
        bad = np.where(nan_ref, ~nan_ok, err > bound)
        if bad.any():
            i = int(np.flatnonzero(bad.reshape(-1))[0])
            failures.append(
                f"{key}: exceeds tolerance at flat {i}: ref {ref_f.reshape(-1)[i]:.6g} "
                f"cand {cand_f.reshape(-1)[i]:.6g} bound {bound.reshape(-1)[i]:.6g}")
    extra = set(cand_out) - set(ref_out)
    if extra:
        failures.append(f"unexpected extra outputs: {sorted(extra)}")
    return (not failures), failures


def evaluate(spec, impl, shape, seed, cand_fn):
    """Full check of one candidate implementation on one (shape, seed).

    cand_fn(**inputs) -> dict of outputs. Returns (ok, failures).
    """
    inputs = impl.gen_inputs(shape, seed)
    ref_out = impl.ref(**inputs)
    cand_out = cand_fn(**inputs)
    ok, failures = check_outputs(spec, impl, inputs, ref_out, cand_out)
    aliased = antialias_violation(inputs, cand_out)
    if aliased:
        ok = False
        failures.append("anti-cheat: output aliases an input array")
    return ok, failures


def antialias_violation(inputs, cand_out):
    """True if any candidate output shares memory with (or is) an input array."""
    for out in cand_out.values():
        if not hasattr(out, "shape"):
            continue
        for inp in inputs.values():
            if isinstance(inp, np.ndarray) and np.shares_memory(out, inp):
                return True
            if isinstance(inp, (int, np.integer)) and isinstance(out, np.ndarray):
                continue
    return False
