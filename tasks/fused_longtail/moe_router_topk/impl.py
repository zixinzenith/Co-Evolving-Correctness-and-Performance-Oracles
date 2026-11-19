"""Task impl: moe_router_topk. ref is ground truth; gen_inputs is the adversarial generator."""

import numpy as np

from tasks.common import gen


def ref(**inp):
    logits, k = inp['logits'], int(inp['k'][0])
    z = logits - logits.max(axis=-1, keepdims=True)
    p = np.exp(z)
    p = (p / p.sum(axis=-1, keepdims=True)).astype(np.float32)
    order = np.argsort(-p, axis=-1, kind='stable')  # ties resolved by lower index first
    idx = order[..., :k].astype(np.int64)
    w = np.take_along_axis(p, idx, axis=-1)
    w = (w / w.sum(axis=-1, keepdims=True)).astype(np.float32)
    return {'probs': p, 'idx': idx, 'w_topk': w}


def gen_inputs(shape, seed):
    M, E = shape["M"], shape["E"]
    logits = gen.logits_near_tie((M, E), seed)
    return {'logits': logits.astype(np.float32), 'k': np.array([int(shape["k"])], np.int64)}
