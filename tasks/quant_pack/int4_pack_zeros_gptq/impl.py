"""Task impl: int4_pack_zeros_gptq. ref is ground truth; gen_inputs is the adversarial generator."""

import numpy as np
import torch

from tasks.common import gen, ops


def ref(**inp):
    return {'qzeros': ops.pack_gptq_qzeros(inp['zeros'])}


def gen_inputs(shape, seed):
    K, N, g, M = shape["K"], shape["N"], shape["g"], shape["M"]
    z = gen.int4_boundary((K // g, N), seed)
    return {'zeros': z}
