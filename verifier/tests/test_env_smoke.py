"""Smoke test: the pinned CPU environment imports and runs torch + numpy."""

import numpy as np
import torch


def test_torch_cpu_only():
    assert not torch.cuda.is_available()
    assert torch.version.cuda is None


def test_torch_numpy_roundtrip():
    a = np.arange(16, dtype=np.int8).reshape(4, 4)
    t = torch.from_numpy(a)
    assert np.array_equal((t * 2).numpy(), a * 2)


def test_float64_matmul_matches_numpy():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((8, 16))
    w = rng.standard_normal((16, 4))
    got = (torch.from_numpy(x) @ torch.from_numpy(w)).numpy()
    np.testing.assert_allclose(got, x @ w, rtol=1e-12, atol=1e-12)
