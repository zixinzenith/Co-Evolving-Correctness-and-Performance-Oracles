"""Task impl: kv_quant_int4_asym_per_head. ref is ground truth; gen_inputs is the adversarial generator."""

import numpy as np

from tasks.common import gen


def ref(**inp):
    k = inp['k'].astype(np.float32)
    B, H, S, D = k.shape
    scales, zeros = inp['scales'], inp['zeros']
    g = D // scales.shape[-1]
    s = np.repeat(scales, g, axis=-1)  # block repeat: group j covers D positions j*g..j*g+g-1
    z = np.repeat(zeros, g, axis=-1)
    s = np.where(s == 0, np.float32(1.0), s)  # zero-group fallback
    q = np.clip(np.rint((k - z) / s), 0, 15)
    return {'q': q.astype(np.int64)}


def gen_inputs(shape, seed):
    B, H, S, D, g = shape["B"], shape["H"], shape["S"], shape["D"], shape["g"]
    k = gen.kv_fp32((B, H, S, D), seed) * 4.0
    kg = k.reshape(B, H, S, D // g, g)
    mn = kg.min(axis=-1, keepdims=True)
    mx = kg.max(axis=-1, keepdims=True)
    s = ((mx - mn) / 15.0).astype(np.float32)
    s = np.where(s == 0, np.float32(1.0), s)  # keeps the zeroed token rows benign
    z = mn.astype(np.float32)
    return {'k': k.astype(np.float32),
            'scales': s.reshape(B, H, S, D // g),
            'zeros': z.reshape(B, H, S, D // g)}
