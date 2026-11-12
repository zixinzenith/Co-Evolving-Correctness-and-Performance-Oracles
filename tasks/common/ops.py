"""Reference primitives for quantized-operator tasks.

Integer packing is done in int64 (numpy) to avoid the int32 << 28 overflow, then cast down.
Layouts follow formats/int4_w4a16_packing.md exactly:
- GPTQ/fbgemm: qweight (K/8, N) packs along K, low nibble first;
  qzeros (K/g, N/8) packs along N with zeros-1 offset.
- AWQ: qweight (K, N/8) and qzeros (K/g, N/8) both pack along N,
  nibble j of word c holds logical column 8c + AWQ_ORDER[j], no zero offset.
"""

import numpy as np
import torch

AWQ_ORDER = [0, 2, 4, 6, 1, 3, 5, 7]


# ---------------------------------------------------------------- GPTQ layout

def pack_gptq_qweight(q: np.ndarray) -> np.ndarray:
    """q: (K, N) int values in [0, 15] (logical, before packing) -> qweight (K/8, N) int32."""
    K, N = q.shape
    assert K % 8 == 0, "K must be divisible by 8"
    q = q.astype(np.int64)
    qw = np.zeros((K // 8, N), dtype=np.int64)
    for j in range(8):
        qw |= q[j::8, :] << (4 * j)
    return qw.astype(np.int32)


def unpack_gptq_qweight(qweight: np.ndarray) -> np.ndarray:
    """qweight (K/8, N) int32 -> q (K, N) int64 (values 0..15)."""
    R, N = qweight.shape
    qw = qweight.astype(np.int64)[:, None, :]  # (R, 1, N)
    shifts = (np.arange(8, dtype=np.int64) * 4)[None, :, None]
    q = (qw >> shifts) & 0xF  # (R, 8, N)
    return q.reshape(R * 8, N)


def pack_gptq_qzeros(zeros: np.ndarray) -> np.ndarray:
    """zeros: (K/g, N) int in [0, 15] -> qzeros (K/g, N/8) int32, with the -1 offset applied."""
    Kg, N = zeros.shape
    assert N % 8 == 0, "N must be divisible by 8"
    z = (zeros.astype(np.int64) - 1) & 0xF  # +1 convention, wrapping mod 16 (logical 0 -> 15)
    qz = np.zeros((Kg, N // 8), dtype=np.int64)
    for j in range(8):
        qz |= z[:, j::8] << (4 * j)
    return qz.astype(np.int32)


def unpack_gptq_qzeros(qzeros: np.ndarray) -> np.ndarray:
    """qzeros (K/g, N/8) int32 -> zeros (K/g, N) int64 in [0, 15] (offset already removed)."""
    Kg, C = qzeros.shape
    qz = qzeros.astype(np.int64)[:, :, None]
    shifts = (np.arange(8, dtype=np.int64) * 4)[None, None, :]
    z = ((qz >> shifts) & 0xF).reshape(Kg, C * 8)
    return (z + 1) & 0xF  # unwrap the +1 convention mod 16


# ---------------------------------------------------------------- AWQ layout

def pack_awq(t: np.ndarray) -> np.ndarray:
    """t: (..., N) int in [0, 15] -> packed (..., N/8) int32, nibble j = column 8c+AWQ_ORDER[j]."""
    N = t.shape[-1]
    assert N % 8 == 0, "N must be divisible by 8"
    t = t.astype(np.int64)
    packed = np.zeros(t.shape[:-1] + (N // 8,), dtype=np.int64)
    for j in range(8):
        packed |= t[..., AWQ_ORDER[j]::8] << (4 * j)
    return packed.astype(np.int32)


def unpack_awq(packed: np.ndarray) -> np.ndarray:
    """packed (..., N/8) int32 -> t (..., N) int64 in [0, 15]."""
    C = packed.shape[-1]
    p = packed.astype(np.int64)[..., None]
    shifts = (np.arange(8, dtype=np.int64) * 4)[None, :]
    nib = (p >> shifts) & 0xF  # (..., C, 8): nib[..., c, j] = value destined for col 8c+ORDER[j]
    out = np.zeros(packed.shape[:-1] + (C * 8,), dtype=np.int64)
    for j in range(8):
        out[..., AWQ_ORDER[j]::8] = nib[..., j]
    return out


# ---------------------------------------------------------------- quantize/dequant

def quantize_int4_asym(w: np.ndarray, scales: np.ndarray, zeros: np.ndarray) -> np.ndarray:
    """w (K, N) fp; scales/zeros (K/g, N) fp -> q (K, N) int64 in [0, 15].

    q = round_half_even((w + zeros*scales) / scales), clipped to [0, 15].
    GPTQ semantics: w ~ (q - zeros) * scales.
    """
    K, N = w.shape
    g = K // scales.shape[0]
    s = np.repeat(scales, g, axis=0)
    z = np.repeat(zeros, g, axis=0)
    q = (w + z * s) / s
    # numpy rint = round-half-to-even, matching torch.round
    q = np.clip(np.rint(q), 0, 15)
    return q.astype(np.int64)


def dequant_int4_group_gptq(qweight, qzeros, scales, out_dtype=np.float32):
    """GPTQ buffers -> w (K, N) float. scales (K/g, N) fp (note: transposed vs AutoGPTQ storage).

    w = (q - zeros) * scales with zeros unpacked from qzeros (+1 removed by unpack).
    """
    q = unpack_gptq_qweight(qweight)
    zeros = unpack_gptq_qzeros(qzeros)
    K, N = q.shape
    g = K // zeros.shape[0]
    s = np.repeat(scales, g, axis=0)
    z = np.repeat(zeros, g, axis=0)
    return ((q - z) * s).astype(out_dtype)


def dequant_int4_group_awq(qweight, qzeros, scales, out_dtype=np.float32):
    """AWQ buffers (qweight (K, N/8), qzeros (K/g, N/8), scales (K/g, N) fp16) -> w (K, N) float."""
    q = unpack_awq(qweight)
    zeros = unpack_awq(qzeros)
    K, N = q.shape
    g = K // zeros.shape[0]
    s = np.repeat(scales, g, axis=0)
    z = np.repeat(zeros, g, axis=0)
    return ((q - z) * s).astype(out_dtype)


def quantize_int8_per_channel(w: np.ndarray):
    """w (K, N) fp -> q int8 (K, N), scales (N,) fp. Symmetric, scale = amax(|w|, K)/127."""
    amax = np.abs(w).max(axis=0)
    scales = amax / 127.0
    scales[scales == 0] = 1.0
    q = np.clip(np.rint(w / scales), -128, 127)
    return q.astype(np.int8), scales


def quantize_int8_per_token(x: np.ndarray):
    """x (M, K) fp -> q int8 (M, K), scales (M, 1) fp. Symmetric per row."""
    amax = np.abs(x).max(axis=1, keepdims=True)
    scales = amax / 127.0
    scales[scales == 0] = 1.0
    q = np.clip(np.rint(x / scales), -128, 127)
    return q.astype(np.int8), scales


def fp8_bytes(x, torch_dtype):
    """float array -> uint8 bit patterns of the fp8 cast (bit-exact ground truth)."""
    t = torch.from_numpy(np.ascontiguousarray(x)).to(torch_dtype)
    return t.view(torch.uint8).numpy()


def fp8_to_float(bits, torch_dtype):
    """uint8 bit patterns -> float32 values (decode only; no rounding involved)."""
    t = torch.from_numpy(np.ascontiguousarray(bits))
    return t.view(torch_dtype).to(torch.float32).numpy()
