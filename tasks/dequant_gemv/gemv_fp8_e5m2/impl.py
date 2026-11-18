"""Task impl: gemv_fp8_e5m2. ref is ground truth; gen_inputs is the adversarial generator."""

import numpy as np
import torch

from tasks.common import gen, ops


def ref(**inp):
    w = ops.fp8_to_float(inp['w_bits'], torch.float8_e5m2) * inp['w_scale'][0]
    y = (w.T @ inp['x']).astype(np.float32)
    return {'y': y}


def tol_scale(**inp):
    """e5m2 carries ~2^-2.5 relative mantissa error per element; coef 0.2 applied by verifier."""
    w = ops.fp8_to_float(inp['w_bits'], torch.float8_e5m2)
    S = (np.abs(w) * np.abs(inp['x'])[:, None]).sum(axis=0)
    return {'y': S.astype(np.float32)}


def gen_inputs(shape, seed):
    K, N = shape["K"], shape["N"]
    w = gen.fp8_adversarial((K, N), seed, 'e5m2') * 0.125
    w_bits = ops.fp8_bytes(w, torch.float8_e5m2)
    x = gen.vec_fp32((K,), seed + 1)
    return {'w_bits': w_bits, 'w_scale': np.array([0.125], np.float32),
            'x': x.astype(np.float32)}
