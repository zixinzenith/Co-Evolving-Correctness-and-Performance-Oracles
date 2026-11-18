"""Task impl: gemm_fp8_e4m3_rowwise. ref is ground truth; gen_inputs is the adversarial generator."""

import numpy as np
import torch

from tasks.common import gen, ops


def ref(**inp):
    a = ops.fp8_to_float(inp['a_bits'], torch.float8_e4m3fn)
    b = ops.fp8_to_float(inp['b_bits'], torch.float8_e4m3fn)
    y = (a * inp['a_scales'][:, None]) @ (b * inp['b_scales'][None, :])
    return {'y': y.astype(np.float32)}


def tol_scale(**inp):
    """e4m3fn mantissa error ~2^-4.5 per element: S_mn = as_m * bs_n * sum_k |a_mk| |b_kn|."""
    a = ops.fp8_to_float(inp['a_bits'], torch.float8_e4m3fn)
    b = ops.fp8_to_float(inp['b_bits'], torch.float8_e4m3fn)
    S = inp['a_scales'][:, None] * inp['b_scales'][None, :] * (np.abs(a) @ np.abs(b))
    return {'y': S.astype(np.float32)}


def gen_inputs(shape, seed):
    K, N, M = shape["K"], shape["N"], shape["M"]
    a = gen.act_fp32((M, K), seed)
    a_bits, a_scales = _cast_rowwise(a)
    b = gen.fp8_adversarial((K, N), seed + 1, 'e4m3')
    b = np.where(np.isfinite(b), b, np.float32(1e4))  # cast-with-scale is undefined at inf/nan
    b_bits, b_scales = _cast_colwise(b)
    return {'a_bits': a_bits, 'a_scales': a_scales.astype(np.float32),
            'b_bits': b_bits, 'b_scales': b_scales.astype(np.float32)}


def _cast_rowwise(a):
    s = np.abs(a).max(axis=1, keepdims=True) / 448.0
    s[s == 0] = 1.0
    bits = ops.fp8_bytes(a / s, torch.float8_e4m3fn)
    return bits, s.astype(np.float32).reshape(-1)


def _cast_colwise(b):
    s = np.abs(b).max(axis=0, keepdims=True) / 448.0
    s[s == 0] = 1.0
    bits = ops.fp8_bytes(b / s, torch.float8_e4m3fn)
    return bits, s.astype(np.float32).reshape(-1)
