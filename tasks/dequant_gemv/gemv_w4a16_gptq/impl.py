"""Task impl: gemv_w4a16_gptq. ref is ground truth; gen_inputs is the adversarial generator."""

import numpy as np

from tasks.common import gen, ops


def ref(**inp):
    w = ops.dequant_int4_group_gptq(inp['qweight'], inp['qzeros'], inp['scales'])
    y = (w.T @ inp['x']).astype(np.float32)
    return {'y': y}


def tol_scale(**inp):
    """Format-derived error scale: half a code step (s/2) per weight, weighted by |x|.

    The verifier accepts |cand - ref| <= coef * S + atol with coef ~ 0.5; anything above
    means the dequant contract (layout, zero offset, group boundaries) was violated.
    """
    scales, x = inp['scales'], inp['x']
    g = inp['x'].shape[0] // scales.shape[0]
    s_rep = np.repeat(scales, g, axis=0)
    S = (s_rep * np.abs(x)[:, None]).sum(axis=0)
    return {'y': S.astype(np.float32)}


def gen_inputs(shape, seed):
    K, N, g = shape["K"], shape["N"], shape["g"]
    q = gen.int4_uniform((K, N), seed)
    z = gen.int4_boundary((K // g, N), seed + 1)
    s = np.abs(gen.fp16_weights((K // g, N), seed + 2)) * 0.01 + 1e-4
    x = gen.vec_fp32((K,), seed + 3)
    return {'qweight': ops.pack_gptq_qweight(q),
            'qzeros': ops.pack_gptq_qzeros(z),
            'scales': s.astype(np.float32),
            'x': x.astype(np.float32)}
