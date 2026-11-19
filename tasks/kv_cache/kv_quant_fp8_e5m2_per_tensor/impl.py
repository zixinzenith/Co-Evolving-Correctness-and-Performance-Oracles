"""Task impl: kv_quant_fp8_e5m2_per_tensor. ref is ground truth; gen_inputs is the adversarial generator."""

import numpy as np
import torch

from tasks.common import gen, ops


def ref(**inp):
    bits = ops.fp8_bytes(inp['k'] / inp['scale'][0], torch.float8_e5m2)
    return {'bits': bits}


def gen_inputs(shape, seed):
    B, H, S, D = shape["B"], shape["H"], shape["S"], shape["D"]
    k = gen.kv_fp32((B, H, S, D), seed) * 30.0  # k/scale spans the subnormal..inf range
    # fixed scale 0.01: k/scale of 57.344 -> max finite, 60 -> rounds to max, > 57.344*1e6-scale -> inf
    specials = np.array([0.0, 1e-5, 0.57344, 0.6, 1.0, 1e4], np.float32)
    if D >= len(specials):
        k[..., 0, 0, :len(specials)] = specials
    return {'k': k.astype(np.float32), 'scale': np.array([0.01], np.float32)}
