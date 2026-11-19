"""Task impl: kv_quant_fp8_e4m3_per_channel. ref is ground truth; gen_inputs is the adversarial generator."""

import numpy as np
import torch

from tasks.common import gen, ops


def ref(**inp):
    k = inp['k']
    amax = np.abs(k).max(axis=-2, keepdims=True)  # over S: per (B, H, 1, D)
    s = (amax / 448.0).astype(np.float32)
    s = np.where(s == 0, np.float32(1.0), s)
    bits = ops.fp8_bytes(k / s, torch.float8_e4m3fn)
    return {'bits': bits, 'scales': s[:, :, 0, :]}


def gen_inputs(shape, seed):
    B, H, S, D = shape["B"], shape["H"], shape["S"], shape["D"]
    k = gen.kv_fp32((B, H, S, D), seed) * 100.0
    k[..., 0] = 0.0  # one channel zero across the cache -> scale fallback
    return {'k': k.astype(np.float32)}
