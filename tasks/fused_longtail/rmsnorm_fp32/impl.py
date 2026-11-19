"""Task impl: rmsnorm_fp32. ref is ground truth; gen_inputs is the adversarial generator."""

import numpy as np

from tasks.common import gen


def ref(**inp):
    x, w = inp['x'], inp['w']
    ms = (x * x).mean(axis=-1, keepdims=True)
    y = x / np.sqrt(ms + 1e-6) * w
    return {'y': y.astype(np.float32)}


def gen_inputs(shape, seed):
    M, K = shape["M"], shape["K"]
    x = gen.act_fp32((M, K), seed)
    w = gen.fp16_weights((K,), seed + 1, outliers=False).astype(np.float32)
    return {'x': x.astype(np.float32), 'w': w}
