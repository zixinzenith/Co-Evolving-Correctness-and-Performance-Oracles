"""Task impl: w4a16_dequant_gptq. ref is ground truth; gen_inputs is the adversarial generator."""

import numpy as np
import torch

from tasks.common import gen, ops


def ref(**inp):
    w = ops.dequant_int4_group_gptq(inp['qweight'], inp['qzeros'], inp['scales'])
    return {'w': w}


def gen_inputs(shape, seed):
    K, N, g, M = shape["K"], shape["N"], shape["g"], shape["M"]
    q = gen.int4_uniform((K, N), seed)
    z = gen.int4_boundary((K // g, N), seed + 1)
    s = np.abs(gen.fp16_weights((K // g, N), seed + 2)) * 0.01 + 1e-4
    return {'qweight': ops.pack_gptq_qweight(q),
            'qzeros': ops.pack_gptq_qzeros(z),
            'scales': s.astype(np.float32)}
