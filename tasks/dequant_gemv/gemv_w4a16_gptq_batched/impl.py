"""Task impl: gemv_w4a16_gptq_batched. ref is ground truth; gen_inputs is the adversarial generator."""

import numpy as np

from tasks.common import gen, ops


def ref(**inp):
    w = ops.dequant_int4_group_gptq(inp['qweight'], inp['qzeros'], inp['scales'])
    y = (inp['x'] @ w).astype(np.float32)
    return {'y': y}


def tol_scale(**inp):
    """Per-row copy of the per-column format bound: S_n = sum_k s_g(k),n |x_mk|, same for all m."""
    scales = inp['scales']
    K = inp['x'].shape[1]
    g = K // scales.shape[0]
    s_rep = np.repeat(scales, g, axis=0)
    S = (s_rep * np.abs(inp['x']).sum(axis=0)[:, None]).sum(axis=0)
    return {'y': S.astype(np.float32)}  # (N,), broadcast over M by the verifier


def gen_inputs(shape, seed):
    K, N, g, M = shape["K"], shape["N"], shape["g"], shape["M"]
    q = gen.int4_uniform((K, N), seed)
    z = gen.int4_boundary((K // g, N), seed + 1)
    s = np.abs(gen.fp16_weights((K // g, N), seed + 2)) * 0.01 + 1e-4
    x = gen.act_fp32((M, K), seed + 3)
    return {'qweight': ops.pack_gptq_qweight(q),
            'qzeros': ops.pack_gptq_qzeros(z),
            'scales': s.astype(np.float32),
            'x': x.astype(np.float32)}
