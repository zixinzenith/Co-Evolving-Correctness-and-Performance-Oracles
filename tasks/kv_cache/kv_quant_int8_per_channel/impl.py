"""Task impl: kv_quant_int8_per_channel. ref is ground truth; gen_inputs is the adversarial generator."""

import numpy as np

from tasks.common import gen


def ref(**inp):
    k = inp['k']
    amax = np.abs(k).max(axis=-2, keepdims=True)  # per (B, H, 1, D): channel-wise over S
    s = (amax / 127.0).astype(np.float32)
    s = np.where(s == 0, np.float32(1.0), s)
    q = np.clip(np.rint(k / s), -128, 127)
    return {'q': q.astype(np.int8), 'scales': s[:, :, 0, :]}


def gen_inputs(shape, seed):
    B, H, S, D = shape["B"], shape["H"], shape["S"], shape["D"]
    k = gen.kv_fp32((B, H, S, D), seed) * 100.0
    k[..., 0] = 0.0  # one channel zero across the whole cache -> scale fallback
    return {'k': k.astype(np.float32)}
