"""Task impl: int4_unpack_awq. ref is ground truth; gen_inputs is the adversarial generator."""

import numpy as np
import torch

from tasks.common import gen, ops


def ref(**inp):
    return {'q': ops.unpack_awq(inp['qweight']),
            'zeros': ops.unpack_awq(inp['qzeros'])}


def gen_inputs(shape, seed):
    K, N, g, M = shape["K"], shape["N"], shape["g"], shape["M"]
    q = gen.int4_boundary((K, N), seed)
    z = gen.int4_uniform((K // g, N), seed + 1)
    return {'qweight': ops.pack_awq(q), 'qzeros': ops.pack_awq(z)}
