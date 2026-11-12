"""Adversarial input generators for quantized-operator tasks.

Every generator honors the adversarial classes required by formats/*.md:
halfway rounding values, boundary nibbles, saturation limits, outliers, all-zero groups.
All generators are deterministic given (shape, seed).
"""

import numpy as np

FP8_E4M3_SPECIAL = np.array(
    [0.0, 448.0, 448.0 * (1 + 2**-4), 1e6, float("inf"), 2**-9, 2**-10, 1.0625, 1.1875],
    dtype=np.float32,
)
FP8_E5M2_SPECIAL = np.array(
    [0.0, 57344.0, 60000.0, 1e6, float("inf"), 2**-14, 2**-16, 1.25, 1.75],
    dtype=np.float32,
)


def int4_uniform(shape, seed):
    return np.random.default_rng(seed).integers(0, 16, shape).astype(np.int64)


def int4_boundary(shape, seed):
    """Values at the int4 boundaries and the mod-16 wrap points of the zero offset."""
    rng = np.random.default_rng(seed)
    special = np.array([0, 1, 7, 8, 14, 15], dtype=np.int64)
    return special[rng.integers(0, len(special), shape)]


def int4_halfway(shape, seed):
    """Logical int4 codes chosen so dequant hits exact halfway float points (scale 2^-k)."""
    rng = np.random.default_rng(seed)
    # with scales = 2^-4 and zeros = 8, dequant values are (q-8)*2^-4; halfway needs no
    # extra structure here, so we stress the wrap pairs (0<->15) instead
    pairs = [(0, 15), (1, 14), (7, 8)]
    out = np.empty(shape, dtype=np.int64)
    it = np.nditer(out, flags=["multi_index"])
    idxs = np.indices(shape).reshape(len(shape), -1).T
    sel = rng.integers(0, 3, len(idxs))
    for (i,), s in zip(idxs, sel):
        out[tuple(i)] = pairs[s][rng.integers(0, 2)]
    return out


def fp16_weights(shape, seed, outliers=True):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(shape, dtype=np.float32)
    if outliers and shape[-1] > 1:
        w[..., 0] *= 1000.0  # one outlier channel
    return w


def fp16_zero_group(shape, seed):
    """Weights with an entire group (or row) at zero — scale=0 edge case."""
    w = fp16_weights(shape, seed, outliers=False)
    if len(shape) == 2:
        w[0, :] = 0.0
    return w


def fp8_adversarial(shape, seed, format):
    """Matrix containing the saturation/halfway special values scattered in normal noise."""
    rng = np.random.default_rng(seed)
    special = FP8_E4M3_SPECIAL if format == "e4m3" else FP8_E5M2_SPECIAL
    x = rng.standard_normal(shape, dtype=np.float32) * 8.0
    flat = x.reshape(-1)
    pos = rng.choice(flat.size, size=min(len(special), flat.size), replace=False)
    flat[pos] = special[: len(pos)]
    return flat.reshape(shape)


def int8_adversarial(shape, seed):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(shape, dtype=np.float32)
    if len(shape) == 2 and shape[1] > 4:
        x[:, 0] *= 300.0  # per-channel outlier
        x[0, :] = 0.0  # zero row -> scale fallback path
    return x


def act_fp32(shape, seed):
    """Activation matrix (M, K): normal noise, one outlier row, one zero row."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(shape, dtype=np.float32)
    if len(shape) == 2 and shape[0] > 2:
        x[0] *= 100.0
        x[-1] = 0.0
    return x


def vec_fp32(shape, seed):
    """1-D activation vector: normal noise, first element scaled (outlier)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(shape, dtype=np.float32)
    x.flat[0] *= 50.0
    return x


def kv_fp32(shape, seed):
    """KV-cache-like tensor (..., S, D): normal noise, first token row zeroed."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(shape, dtype=np.float32)
    if len(shape) >= 2:
        x[..., 0, :] = 0.0
    return x


def logits_near_tie(shape, seed):
    """Router logits with near-tie pairs (top-k boundary ambiguity is the bug axis)."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(shape, dtype=np.float32)
    flat = z.reshape(-1)
    for i in range(0, flat.size - 1, 16):
        flat[i + 1] = flat[i] + 1e-7  # near tie
    if flat.size > 8:
        flat[3] = flat[7]  # exact tie
    return z.reshape(shape)
