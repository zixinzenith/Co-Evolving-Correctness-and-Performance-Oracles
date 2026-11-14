"""Task impl: int8_quant_per_channel. ref is ground truth; gen_inputs is the adversarial generator."""

import numpy as np
import torch

from tasks.common import gen, ops


def ref(**inp):
    q, s = ops.quantize_int8_per_channel(inp['w'])
    return {'q': q, 'scales': s}


def gen_inputs(shape, seed):
    K, N, g, M = shape["K"], shape["N"], shape["g"], shape["M"]
    w = gen.int8_adversarial((K, N), seed)
    return {'w': w}
