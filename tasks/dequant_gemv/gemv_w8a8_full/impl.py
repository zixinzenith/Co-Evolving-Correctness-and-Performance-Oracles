"""Task impl: gemv_w8a8_full. ref is ground truth; gen_inputs is the adversarial generator."""

import numpy as np

from tasks.common import gen


def ref(**inp):
    acc = inp['x_q'].astype(np.int64) @ inp['w_q'].astype(np.int64)  # exact int accumulation
    y = acc.astype(np.float32) * inp['x_scales'][:, None] * inp['w_scales'][None, :]
    return {'y': y.astype(np.float32)}


def tol_scale(**inp):
    """Activation and weight each carry up to half a quantization step of error:
    S_mn = sx_m * sw_n * (sum_k |x_q_mk| + sum_k |w_q_kn|) / 1 (coef ~0.5 applied by verifier)."""
    sx, sw = inp['x_scales'], inp['w_scales']
    S = sx[:, None] * sw[None, :] * (
        np.abs(inp['x_q']).sum(axis=1, keepdims=True) + np.abs(inp['w_q']).sum(axis=0)[None, :]
    )
    return {'y': S.astype(np.float32)}


def gen_inputs(shape, seed):
    K, N, M = shape["K"], shape["N"], shape["M"]
    x = gen.act_fp32((M, K), seed)
    x_q, x_scales = _quant_per_token(x)
    w = gen.int8_adversarial((K, N), seed + 1)
    w_q, w_scales = _quant_per_channel(w)
    return {'x_q': x_q, 'x_scales': x_scales.astype(np.float32),
            'w_q': w_q, 'w_scales': w_scales.astype(np.float32)}


def _quant_per_token(x):
    amax = np.abs(x).max(axis=1, keepdims=True)
    s = amax / 127.0
    s[s == 0] = 1.0
    q = np.clip(np.rint(x / s), -128, 127)
    return q.astype(np.int8), s.astype(np.float32).reshape(-1)


def _quant_per_channel(w):
    amax = np.abs(w).max(axis=0, keepdims=True)
    s = amax / 127.0
    s[s == 0] = 1.0
    q = np.clip(np.rint(w / s), -128, 127)
    return q.astype(np.int8), s.astype(np.float32).reshape(-1)
