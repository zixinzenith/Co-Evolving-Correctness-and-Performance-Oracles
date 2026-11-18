"""Task impl: gemv_w4a16_sym_gptq. ref is ground truth; gen_inputs is the adversarial generator."""

import numpy as np

from tasks.common import gen, ops


def ref(**inp):
    q = ops.unpack_gptq_qweight(inp['qweight']).astype(np.float32)
    K, N = q.shape
    g = K // inp['scales'].shape[0]
    s = np.repeat(inp['scales'], g, axis=0)
    w = (q - 8.0) * s  # symmetric: implied zero point is the midpoint 8
    y = (w.T @ inp['x']).astype(np.float32)
    return {'y': y}


def tol_scale(**inp):
    scales, x = inp['scales'], inp['x']
    g = inp['x'].shape[0] // scales.shape[0]
    s_rep = np.repeat(scales, g, axis=0)
    S = (s_rep * np.abs(x)[:, None]).sum(axis=0)
    return {'y': S.astype(np.float32)}


def gen_inputs(shape, seed):
    K, N, g = shape["K"], shape["N"], shape["g"]
    q = gen.int4_boundary((K, N), seed) if seed % 3 == 0 else gen.int4_uniform((K, N), seed)
    s = np.abs(gen.fp16_weights((K // g, N), seed + 1)) * 0.01 + 1e-4
    x = gen.vec_fp32((K,), seed + 2)
    return {'qweight': ops.pack_gptq_qweight(q),
            'scales': s.astype(np.float32),
            'x': x.astype(np.float32)}
