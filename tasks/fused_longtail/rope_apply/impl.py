"""Task impl: rope_apply. ref is ground truth; gen_inputs is the adversarial generator."""

import numpy as np

from tasks.common import gen


def ref(**inp):
    q, cos, sin = inp['q'], inp['cos'], inp['sin']
    qe, qo = q[:, 0::2], q[:, 1::2]
    out = np.empty_like(q)
    out[:, 0::2] = qe * cos - qo * sin
    out[:, 1::2] = qo * cos + qe * sin
    return {'out': out.astype(np.float32)}


def gen_inputs(shape, seed):
    S, D = shape["S"], shape["D"]
    q = gen.fp16_weights((S, D), seed).astype(np.float32)
    i = np.arange(D // 2, dtype=np.float32)
    theta = np.arange(S, dtype=np.float32)[:, None] / (10000.0 ** (2 * i / D))
    cos = np.cos(theta).astype(np.float32)
    sin = np.sin(theta).astype(np.float32)
    return {'q': q, 'cos': cos, 'sin': sin}
