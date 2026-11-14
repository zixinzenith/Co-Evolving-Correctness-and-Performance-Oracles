"""Task impl: fp8_cast_e4m3. ref is ground truth; gen_inputs is the adversarial generator."""

import numpy as np
import torch

from tasks.common import gen, ops


def ref(**inp):
    return {'bits': ops.fp8_bytes(inp['x'], torch.float8_e4m3fn)}


def gen_inputs(shape, seed):
    K, N, g, M = shape["K"], shape["N"], shape["g"], shape["M"]
    x = gen.fp8_adversarial((M, N), seed, 'e4m3')
    return {'x': x}
