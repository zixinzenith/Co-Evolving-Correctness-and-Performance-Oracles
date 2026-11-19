"""Task impl: silu_mul_quant_int8. ref is ground truth; gen_inputs is the adversarial generator."""

import numpy as np

from tasks.common import gen


def ref(**inp):
    a, b = inp['a'], inp['b']
    with np.errstate(over='ignore'):  # exp(-a) overflows to inf for very negative a; silu -> 0
        p = (a / (1.0 + np.exp(-a)) * b).astype(np.float32)
    amax = np.abs(p).max(axis=-1, keepdims=True)
    s = (amax / 127.0).astype(np.float32)
    s = np.where(s == 0, np.float32(1.0), s)
    q = np.clip(np.rint(p / s), -128, 127)
    return {'q': q.astype(np.int8), 's': s[..., 0]}


def gen_inputs(shape, seed):
    M, K = shape["M"], shape["K"]
    a = gen.act_fp32((M, K), seed) * 2.0
    b = gen.fp16_weights((M, K), seed + 1).astype(np.float32)
    return {'a': a.astype(np.float32), 'b': b}
