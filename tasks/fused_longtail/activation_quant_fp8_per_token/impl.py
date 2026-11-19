"""Task impl: activation_quant_fp8_per_token. ref is ground truth; gen_inputs is the adversarial generator."""

import numpy as np
import torch

from tasks.common import gen, ops


def ref(**inp):
    x = inp['x']
    amax = np.abs(x).max(axis=-1, keepdims=True)
    s = (amax / 448.0).astype(np.float32)
    s = np.where(s == 0, np.float32(1.0), s)
    bits = ops.fp8_bytes(x / s, torch.float8_e4m3fn)
    return {'bits': bits, 's': s[..., 0]}


def gen_inputs(shape, seed):
    M, K = shape["M"], shape["K"]
    x = gen.act_fp32((M, K), seed) * 100.0  # amax element lands exactly on the 448 boundary
    if M > 2:
        x[1] = 0.0           # zero row -> scale fallback
        x[-1] = np.float32(1e-3)  # subnormal-range row
    return {'x': x.astype(np.float32)}
