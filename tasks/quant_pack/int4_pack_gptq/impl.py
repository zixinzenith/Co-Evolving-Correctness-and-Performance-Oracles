"""Task impl: int4_pack_gptq. ref is ground truth; gen_inputs is the adversarial generator."""

import numpy as np
import torch

from tasks.common import gen, ops


def ref(**inp):
    return {'qweight': ops.pack_gptq_qweight(inp['q'])}


def gen_inputs(shape, seed):
    K, N, g, M = shape["K"], shape["N"], shape["g"], shape["M"]
    q = gen.int4_boundary((K, N), seed) if seed % 2 else gen.int4_uniform((K, N), seed)
    return {'q': q}
