"""Task impl: layernorm_residual_quant_int8. ref is ground truth; gen_inputs is the adversarial generator."""

import numpy as np

from tasks.common import gen


def ref(**inp):
    res_out = (inp['x'] + inp['res']).astype(np.float32)
    mu = res_out.mean(axis=-1, keepdims=True)
    var = res_out.var(axis=-1, keepdims=True)  # biased variance, as in torch.layer_norm
    y = (res_out - mu) / np.sqrt(var + 1e-5) * inp['w'] + inp['b']
    amax = np.abs(y).max(axis=-1, keepdims=True)
    s = (amax / 127.0).astype(np.float32)
    s = np.where(s == 0, np.float32(1.0), s)
    q = np.clip(np.rint(y / s), -128, 127)
    return {'q': q.astype(np.int8), 's': s[..., 0], 'res_out': res_out}


def gen_inputs(shape, seed):
    M, K = shape["M"], shape["K"]
    x = gen.act_fp32((M, K), seed)
    res = gen.act_fp32((M, K), seed + 1) * 0.1
    w = gen.fp16_weights((K,), seed + 2, outliers=False).astype(np.float32)
    b = gen.fp16_weights((K,), seed + 3, outliers=False).astype(np.float32) * 0.1
    return {'x': x.astype(np.float32), 'res': res.astype(np.float32),
            'w': w, 'b': b}
