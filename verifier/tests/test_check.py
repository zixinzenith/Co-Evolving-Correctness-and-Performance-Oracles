"""Unit tests for the verifier itself: bit-exact checks, derived tolerances, anti-cheat.

The oracle is the load-bearing component of the benchmark, so it gets its own tests —
a wrong verifier would silently accept wrong kernels (the exact failure mode this
benchmark exists to study).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from verifier import anticheat, check  # noqa: E402


def _load(group, name):
    return check.load_task(group, name)


def test_bit_exact_pass():
    spec, impl = _load("quant_pack", "int8_quant_per_channel")
    shape = spec["shapes"]["visible"][0]
    ok, failures = check.evaluate(spec, impl, shape, 0, impl.ref)
    assert ok, failures


def test_bit_exact_single_byte_flip_fails():
    spec, impl = _load("quant_pack", "int8_quant_per_channel")
    inputs = impl.gen_inputs(spec["shapes"]["visible"][0], 0)
    ref_out = impl.ref(**inputs)
    cand = {k: v.copy() for k, v in ref_out.items()}
    cand["q"].reshape(-1)[7] ^= 1
    ok, failures = check.check_outputs(spec, impl, inputs, ref_out, cand)
    assert not ok
    assert any("q" in f for f in failures)


def test_sum_abs_tolerance_band():
    spec, impl = _load("dequant_gemv", "gemv_w4a16_gptq")
    inputs = impl.gen_inputs(spec["shapes"]["visible"][0], 0)
    ref_out = impl.ref(**inputs)
    S = impl.tol_scale(**inputs)["y"]

    inside = {"y": ref_out["y"] + 0.4 * S}   # tol = 0.51*S + atol
    ok, failures = check.check_outputs(spec, impl, inputs, ref_out, inside)
    assert ok, failures

    outside = {"y": ref_out["y"] + 0.6 * S}
    ok, failures = check.check_outputs(spec, impl, inputs, ref_out, outside)
    assert not ok
    assert any("exceeds tolerance" in f for f in failures)


def test_int_shift_window():
    spec, impl = _load("fused_longtail", "layernorm_residual_quant_int8")
    inputs = impl.gen_inputs(spec["shapes"]["visible"][0], 0)
    ref_out = impl.ref(**inputs)

    cand = dict(ref_out)
    cand["q"] = ref_out["q"].astype(np.int16) + 1  # reduction-order flip: within ±1
    ok, _ = check.check_outputs(spec, impl, inputs, ref_out, cand)
    assert ok

    cand["q"] = ref_out["q"].astype(np.int16) + 2  # real contract error
    ok, failures = check.check_outputs(spec, impl, inputs, ref_out, cand)
    assert not ok
    assert any("beyond" in f for f in failures)


def test_shape_mismatch_reported():
    spec, impl = _load("gemm", "gemm_w8a8_int32")
    inputs = impl.gen_inputs(spec["shapes"]["visible"][0], 0)
    ref_out = impl.ref(**inputs)
    cand = {"acc": ref_out["acc"].T.copy()}
    ok, failures = check.check_outputs(spec, impl, inputs, ref_out, cand)
    assert not ok
    assert any("shape" in f for f in failures)


def test_anticheat_import_scan():
    bad = anticheat.scan_source("import numpy as np\nfrom tasks.common import ops\n")
    assert "tasks.common" in bad
    clean = anticheat.scan_source("import numpy as np\nimport torch\n")
    assert clean == []
    sneaky = anticheat.scan_source("import subprocess")
    assert "subprocess" in sneaky


def test_anticheat_aliasing():
    spec, impl = _load("quant_pack", "int8_quant_per_channel")
    inputs = impl.gen_inputs(spec["shapes"]["visible"][0], 0)

    def cheater(**inp):
        return {"q": inp["w"].view(np.int8), "scales": np.ones(inp["w"].shape[1], np.float32)}

    ok, failures = check.evaluate(spec, impl, spec["shapes"]["visible"][0], 0, cheater)
    assert not ok
    assert any("aliases" in f for f in failures)
