"""Task impl: int4_quantize_asym. ref is ground truth; gen_inputs is the adversarial generator."""

import numpy as np
import torch

from tasks.common import gen, ops


def ref(**inp):
    return {'q': ops.quantize_int4_asym(inp['w'], inp['scales'], inp['zeros'])}


def gen_inputs(shape, seed):
    K, N, g, M = shape["K"], shape["N"], shape["g"], shape["M"]
    w = gen.fp16_weights((K, N), seed)
    scales = np.full((K // g, N), 0.0625, dtype=np.float32)
    zeros = np.tile(np.arange(8, dtype=np.float32), (K // g, N // 8))
    if seed % 3 == 1:  # halfway class: (w/s + z) lands exactly on k + 0.5 in fp32
        k = np.arange(K * N, dtype=np.float32).reshape(K, N) % 7
        w = ((k + 0.5) - 8.0) * 0.0625
    return {'w': w, 'scales': scales, 'zeros': zeros}
