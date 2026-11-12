"""Cross-validation of tasks/common/ops.py against the real AutoAWQ semantics.

AutoAWQ's packing_utils.py is vendored at tests/vendor/awq_packing_utils.py (sha256-pinned in
formats/int4_w4a16_packing.md); these tests prove our independent reimplementation of the AWQ
layout matches theirs bit-for-bit, and that GPTQ pack/unpack round-trips.
Run from repo root: make test
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from tasks.common import ops  # noqa: E402

VENDOR = os.path.join(os.path.dirname(__file__), "vendor", "awq_packing_utils.py")


def test_gptq_pack_roundtrip():
    rng = np.random.default_rng(0)
    for K, N in [(64, 16), (128, 64), (256, 8)]:
        q = rng.integers(0, 16, (K, N))
        back = ops.unpack_gptq_qweight(ops.pack_gptq_qweight(q))
        np.testing.assert_array_equal(back, q)
        z = rng.integers(0, 16, (K // 32, N))
        np.testing.assert_array_equal(
            ops.unpack_gptq_qzeros(ops.pack_gptq_qzeros(z)), z
        )


def test_gptq_nibble_order():
    # nibble j of word row r holds K-position 8r + j (low nibble first)
    q = np.zeros((8, 1), dtype=np.int64)
    q[3, 0] = 5  # K-position 3 -> nibble 3
    qw = ops.pack_gptq_qweight(q)
    assert qw.shape == (1, 1)
    assert int(qw[0, 0]) == (5 << 12)


def test_gptq_zeros_offset():
    # logical 1 -> stored nibble 0; logical 0 -> stored nibble 15 (wraps mod 16)
    z = np.array([[1, 0, 2, 15, 7, 8, 0, 1]], dtype=np.int64)
    qz = ops.pack_gptq_qzeros(z)
    back = ops.unpack_gptq_qzeros(qz)
    np.testing.assert_array_equal(back, z)
    nib = (int(qz[0, 0]) & 0xFF)  # nibbles 0 (low) and 1
    assert nib == 0xF0  # logical 1 -> 0x0, logical 0 -> 0xF


def test_awq_matches_autoawq():
    if not os.path.exists(VENDOR):
        import pytest

        pytest.skip("vendored AutoAWQ packing_utils not present")
    sys.path.insert(0, os.path.dirname(VENDOR))
    import awq_packing_utils as vendor  # noqa: E402

    rng = np.random.default_rng(1)
    for K, N, g in [(16, 16, 8), (64, 8, 32), (128, 64, 128)]:
        q = rng.integers(0, 16, (K, N))
        z = rng.integers(0, 16, (K // g, N))
        qw = torch.from_numpy(ops.pack_awq(q))
        qz = torch.from_numpy(ops.pack_awq(z))
        iw, iz = vendor.unpack_awq(qw, qz, 4)
        iw, iz = vendor.reverse_awq_order(iw, iz, 4)
        np.testing.assert_array_equal((iw & 0xF).numpy(), q)
        np.testing.assert_array_equal((iz & 0xF).numpy(), z)


def test_awq_dequant_matches_autoawq():
    if not os.path.exists(VENDOR):
        import pytest

        pytest.skip("vendored AutoAWQ packing_utils not present")
    sys.path.insert(0, os.path.dirname(VENDOR))
    import awq_packing_utils as vendor  # noqa: E402

    rng = np.random.default_rng(2)
    K, N, g = 64, 32, 32
    q = rng.integers(0, 16, (K, N))
    z = rng.integers(0, 16, (K // g, N))
    s = (rng.random((K // g, N)) * 0.01 + 0.001).astype(np.float16)
    w_ours = ops.dequant_int4_group_awq(
        ops.pack_awq(q), ops.pack_awq(z), s.astype(np.float32)
    )
    w_vendor = vendor.dequantize_gemm(
        torch.from_numpy(ops.pack_awq(q)),
        torch.from_numpy(ops.pack_awq(z)),
        torch.from_numpy(s),
        4,
        g,
    ).numpy()
    np.testing.assert_allclose(w_ours, w_vendor, rtol=1e-3, atol=1e-5)


def test_fp8_saturation_behavior():
    x = np.array([448.0, 469.0, 1e6, float("inf"), float("nan")], dtype=np.float32)
    b = ops.fp8_bytes(x, torch.float8_e4m3fn)
    vals = torch.from_numpy(b).view(torch.float8_e4m3fn).float()
    assert vals[0] == 448 and vals[1] == 448 and vals[2] == 448 and vals[3] == 448
    assert torch.isnan(vals[4])
    x2 = np.array([57344.0, 60000.0, 1e6, float("inf")], dtype=np.float32)
    v2 = torch.from_numpy(ops.fp8_bytes(x2, torch.float8_e5m2)).view(torch.float8_e5m2).float()
    assert v2[0] == 57344 and v2[1] == 57344 and v2[2] == float("inf") and v2[3] == float("inf")


def test_quantize_half_even():
    # exact halfway values must round to even
    w = np.array([[0.5, 1.5, 2.5, -0.5]], dtype=np.float32)
    s = np.ones((1, 4), dtype=np.float32)
    z = np.zeros((1, 4), dtype=np.float32)
    q = ops.quantize_int4_asym(w, s, z)
    np.testing.assert_array_equal(q, [[0, 2, 2, 0]])
