"""Task impl: int4_convert_gptq_to_awq. ref is ground truth; gen_inputs is the adversarial generator."""

import numpy as np
import torch

from tasks.common import gen, ops


def ref(**inp):
    q = ops.unpack_gptq_qweight(inp['qweight'])
    z = ops.unpack_gptq_qzeros(inp['qzeros'])
    return {'qweight': ops.pack_awq(q), 'qzeros': ops.pack_awq(z)}


def gen_inputs(shape, seed):
    K, N, g, M = shape["K"], shape["N"], shape["g"], shape["M"]
    q = gen.int4_uniform((K, N), seed)
    z = gen.int4_boundary((K // g, N), seed + 1)
    return {'qweight': ops.pack_gptq_qweight(q), 'qzeros': ops.pack_gptq_qzeros(z)}
