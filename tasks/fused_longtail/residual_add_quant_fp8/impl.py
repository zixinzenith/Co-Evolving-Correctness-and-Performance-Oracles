"""Task impl: residual_add_quant_fp8. ref is ground truth; gen_inputs is the adversarial generator."""

import numpy as np
import torch

from tasks.common import gen, ops


def ref(**inp):
    y = (inp['x'] + inp['res']) / inp['scale'][0]
    return {'bits': ops.fp8_bytes(y, torch.float8_e4m3fn)}


def gen_inputs(shape, seed):
    M, K = shape["M"], shape["K"]
    x = gen.act_fp32((M, K), seed) * 10.0
    res = gen.act_fp32((M, K), seed + 1) * 10.0
    # scale 0.05: values of 22.4 land on the 448 boundary, |x+res| > 22.4 saturates
    if K >= 8:
        x[0, :8] = np.array([0.0, 1e-3, 11.2, -11.2, 22.4, 30.0, -22.4, 1.07], np.float32)
        res[0, :8] = 0.0
    return {'x': x.astype(np.float32), 'res': res.astype(np.float32),
            'scale': np.array([0.05], np.float32)}
