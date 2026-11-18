"""Task impl: gemv_w8a8_weight_only. ref is ground truth; gen_inputs is the adversarial generator."""

import numpy as np

from tasks.common import gen


def ref(**inp):
    w = inp['w_q'].astype(np.float32) * inp['w_scales'][None, :]
    y = (w.T @ inp['x']).astype(np.float32)
    return {'y': y}


def tol_scale(**inp):
    """Half a quantization step (s_n/2) per weight, weighted by |x|: S_n = s_n * sum_k |x_k|."""
    S = np.abs(inp['x']).sum() * inp['w_scales']
    return {'y': S.astype(np.float32)}


def gen_inputs(shape, seed):
    K, N = shape["K"], shape["N"]
    w = gen.int8_adversarial((K, N), seed)
    w_q, w_scales = _quant_per_channel(w)
    x = gen.vec_fp32((K,), seed + 1)
    return {'w_q': w_q, 'w_scales': w_scales.astype(np.float32), 'x': x.astype(np.float32)}


def _quant_per_channel(w):
    amax = np.abs(w).max(axis=0)
    s = amax / 127.0
    s[s == 0] = 1.0
    q = np.clip(np.rint(w / s), -128, 127)
    return q.astype(np.int8), s.astype(np.float32)
