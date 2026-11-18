"""Task impl: gemm_fp8_e4m3_e5m2_mixed. ref is ground truth; gen_inputs is the adversarial generator."""

import numpy as np
import torch

from tasks.common import gen, ops


def ref(**inp):
    a = ops.fp8_to_float(inp['a_bits'], torch.float8_e4m3fn) * inp['a_scale'][0]
    b = ops.fp8_to_float(inp['b_bits'], torch.float8_e5m2) * inp['b_scale'][0]
    y = (a @ b).astype(np.float32)
    return {'y': y}


def tol_scale(**inp):
    """Tolerance dominated by the e5m2 operand (~2^-2.5): coef 0.21 on |a| @ |b|, scaled."""
    a = ops.fp8_to_float(inp['a_bits'], torch.float8_e4m3fn)
    b = ops.fp8_to_float(inp['b_bits'], torch.float8_e5m2)
    S = inp['a_scale'][0] * inp['b_scale'][0] * (np.abs(a) @ np.abs(b))
    return {'y': S.astype(np.float32)}


def gen_inputs(shape, seed):
    K, N, M = shape["K"], shape["N"], shape["M"]
    a = gen.fp8_adversarial((M, K), seed, 'e4m3')
    b = gen.fp8_adversarial((K, N), seed + 1, 'e5m2')
    a_bits = ops.fp8_bytes(a * 0.5, torch.float8_e4m3fn)
    b_bits = ops.fp8_bytes(b * 0.5, torch.float8_e5m2)
    return {'a_bits': a_bits, 'a_scale': np.array([0.5], np.float32),
            'b_bits': b_bits, 'b_scale': np.array([0.5], np.float32)}
