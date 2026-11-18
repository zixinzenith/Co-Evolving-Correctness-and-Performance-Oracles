"""Task impl: gemm_w4a16_awq. ref is ground truth; gen_inputs is the adversarial generator."""

import numpy as np

from tasks.common import gen, ops


def ref(**inp):
    w = ops.dequant_int4_group_awq(inp['qweight'], inp['qzeros'], inp['scales'])
    y = (inp['x'] @ w).astype(np.float32)
    return {'y': y}


def tol_scale(**inp):
    scales, x = inp['scales'], inp['x']
    g = x.shape[1] // scales.shape[0]
    s_rep = np.repeat(scales, g, axis=0)
    S = (s_rep[None, :, :] * np.abs(x)[:, :, None]).sum(axis=1)
    return {'y': S.astype(np.float32)}


def gen_inputs(shape, seed):
    K, N, g, M = shape["K"], shape["N"], shape["g"], shape["M"]
    q = gen.int4_boundary((K, N), seed) if seed % 2 else gen.int4_uniform((K, N), seed)
    z = gen.int4_uniform((K // g, N), seed + 1)
    s = np.abs(gen.fp16_weights((K // g, N), seed + 2)) * 0.01 + 1e-4
    x = gen.act_fp32((M, K), seed + 3)
    return {'qweight': ops.pack_awq(q),
            'qzeros': ops.pack_awq(z),
            'scales': s.astype(np.float32),
            'x': x.astype(np.float32)}
