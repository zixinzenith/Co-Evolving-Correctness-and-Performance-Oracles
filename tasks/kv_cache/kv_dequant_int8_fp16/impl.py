"""Task impl: kv_dequant_int8_fp16. ref is ground truth; gen_inputs is the adversarial generator."""

import numpy as np

from tasks.common import gen


def ref(**inp):
    prod = inp['q'].astype(np.float16) * inp['scales'][..., None]  # fp16 storage arithmetic
    return {'out': prod.astype(np.float32)}


def gen_inputs(shape, seed):
    B, H, S, D = shape["B"], shape["H"], shape["S"], shape["D"]
    rng = np.random.default_rng(seed)
    q = rng.integers(-128, 128, (B, H, S, D)).astype(np.int8)
    s = (rng.random((B, H, S)) * 0.05 + 1e-3).astype(np.float16)
    s[..., 0] = np.float16(0.0)  # zero token row: q * 0 == 0 exactly, no fallback here
    return {'q': q, 'scales': s}
