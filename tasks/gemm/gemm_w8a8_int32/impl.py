"""Task impl: gemm_w8a8_int32. ref is ground truth; gen_inputs is the adversarial generator."""

import numpy as np

from tasks.common import gen


def ref(**inp):
    acc = inp['x_q'].astype(np.int64) @ inp['w_q'].astype(np.int64)
    return {'acc': acc.astype(np.int32)}


def gen_inputs(shape, seed):
    K, N, M = shape["K"], shape["N"], shape["M"]
    x = gen.act_fp32((M, K), seed)
    w = gen.int8_adversarial((K, N), seed + 1)
    return {'x_q': _quant(x, axis=1), 'w_q': _quant(w, axis=0)}


def _quant(a, axis):
    amax = np.abs(a).max(axis=axis, keepdims=True)
    s = amax / 127.0
    s[s == 0] = 1.0
    return np.clip(np.rint(a / s), -128, 127).astype(np.int8)
