"""Hand-written faulty variants of the reference kernels (mutants).

Each mutant plants ONE documented bug axis from formats/rounding_and_scale_conventions.md
into a copy of the reference computation: wrong nibble order, missing zero offset, half-up
rounding, wrong scale broadcast, wrong quantization axis, fp8 format confusion, dropped
fallbacks, wrong reduction semantics, tie-breaking order.

mutants/run_mutants.py feeds every mutant to the verifier and reports the detection rate.
A mutant that escapes means the oracle is too weak for that fault class — the exact
oracle-adequacy signal KernelBench-M (arXiv:2609.22220) formalized and this benchmark's
verifier is designed to close.
"""

import numpy as np
import torch

from tasks.common import ops


MUTANTS = {}  # (group, task) -> [(mutant_name, fn(**inp) -> dict)]


def mutant(group, task, name):
    def deco(fn):
        MUTANTS.setdefault((group, task), []).append((name, fn))
        return fn
    return deco


# ---------------------------------------------------------------- quant_pack (A)

@mutant("quant_pack", "int4_pack_awq", "identity_nibble_order")
def _(inp):
    """Packs along N but with the identity nibble order instead of the AWQ permutation."""
    q = inp["q"].astype(np.int64)
    N = q.shape[-1]
    packed = np.zeros(q.shape[:-1] + (N // 8,), np.int64)
    for j in range(8):
        packed |= q[..., j::8] << (4 * j)
    return {"qweight": packed.astype(np.int32),
            "qzeros": ops.pack_awq(inp["zeros"])}


@mutant("quant_pack", "int4_pack_gptq", "high_nibble_first")
def _(inp):
    """GPTQ qweight with nibble j shifted to 4*(7-j) instead of 4*j."""
    q = inp["q"].astype(np.int64)
    K, N = q.shape
    qw = np.zeros((K // 8, N), np.int64)
    for j in range(8):
        qw |= q[j::8, :] << (4 * (7 - j))
    return {"qweight": qw.astype(np.int32)}


@mutant("quant_pack", "int4_pack_zeros_gptq", "missing_minus1_offset")
def _(inp):
    """Packs zero-points without the -1 offset (reserves no all-zero code)."""
    z = inp["zeros"].astype(np.int64) & 0xF
    Kg, N = z.shape
    qz = np.zeros((Kg, N // 8), np.int64)
    for j in range(8):
        qz |= z[:, j::8] << (4 * j)
    return {"qzeros": qz.astype(np.int32)}


@mutant("quant_pack", "int4_unpack_gptq", "missing_plus1_unwrap")
def _(inp):
    """Unpacks qzeros without removing the +1 convention."""
    qz = inp["qzeros"].astype(np.int64)
    Kg, C = qz.shape
    shifts = (np.arange(8, dtype=np.int64) * 4)[None, None, :]
    z = ((qz[:, :, None] >> shifts) & 0xF).reshape(Kg, C * 8)
    return {"zeros": z}


@mutant("quant_pack", "int4_unpack_awq", "no_order_reverse")
def _(inp):
    """Unpacks AWQ buffers into nibble order without applying AWQ_ORDER."""
    p = inp["qweight"].astype(np.int64)
    C = p.shape[-1]
    shifts = (np.arange(8, dtype=np.int64) * 4)[None, :]
    nib = (p[..., None] >> shifts) & 0xF  # (..., C, 8)
    return {"q": nib.reshape(p.shape[:-1] + (C * 8,)),
            "zeros": ops.unpack_awq(inp["qzeros"])}


@mutant("quant_pack", "int4_quantize_asym", "half_up_rounding")
def _(inp):
    """Quantizes with floor(x + 0.5) instead of round-half-to-even."""
    w, s0, z0 = inp["w"], inp["scales"], inp["zeros"]
    g = w.shape[0] // s0.shape[0]
    s = np.repeat(s0, g, axis=0)
    z = np.repeat(z0, g, axis=0)
    q = np.clip(np.floor((w + z * s) / s + 0.5), 0, 15)
    return {"q": q.astype(np.int64)}


@mutant("quant_pack", "int8_quant_per_channel", "half_up_rounding")
def _(inp):
    """Symmetric per-channel INT8 with floor(x + 0.5) rounding."""
    w = inp["w"]
    amax = np.abs(w).max(axis=0)
    scales = amax / 127.0
    scales = np.where(scales == 0, 1.0, scales)
    q = np.clip(np.floor(w / scales + 0.5), -128, 127)
    return {"q": q.astype(np.int8), "scales": scales.astype(np.float32)}


@mutant("quant_pack", "int8_quant_per_token", "scale_on_wrong_axis")
def _(inp):
    """Per-row quantization accidentally done per-column (channel axis)."""
    x = inp["x"]
    amax = np.abs(x).max(axis=0, keepdims=True)  # wrong: max over rows
    scales = np.broadcast_to(amax / 127.0, (x.shape[1],)).copy()
    scales[scales == 0] = 1.0
    q = np.clip(np.rint(x / scales), -128, 127)
    return {"q": q.astype(np.int8), "scales": scales.astype(np.float32)}


@mutant("quant_pack", "fp8_cast_e4m3", "cast_as_e5m2")
def _(inp):
    """Casts to the wrong fp8 format (e5m2 preserves inf; e4m3fn saturates at 448)."""
    return {"bits": ops.fp8_bytes(inp["x"], torch.float8_e5m2)}


@mutant("quant_pack", "fp8_cast_e5m2", "cast_as_e4m3")
def _(inp):
    """Casts to the wrong fp8 format (saturates at 448 instead of preserving inf)."""
    return {"bits": ops.fp8_bytes(inp["x"], torch.float8_e4m3fn)}


@mutant("quant_pack", "int4_convert_gptq_to_awq", "no_zero_unwrap")
def _(inp):
    """Converts layouts but skips removing the GPTQ +1 zero offset on the way."""
    q = ops.unpack_gptq_qweight(inp["qweight"])
    qz = inp["qzeros"].astype(np.int64)
    Kg, C = qz.shape
    shifts = (np.arange(8, dtype=np.int64) * 4)[None, None, :]
    z = ((qz[:, :, None] >> shifts) & 0xF).reshape(Kg, C * 8)  # offset not removed
    return {"qweight": ops.pack_awq(q), "qzeros": ops.pack_awq(z)}


@mutant("quant_pack", "w4a16_dequant_gptq", "tile_instead_of_repeat")
def _(inp):
    """Dequantizes with np.tile scales — tile order differs from repeat (group) order."""
    q = ops.unpack_gptq_qweight(inp["qweight"]).astype(np.float32)
    zeros = ops.unpack_gptq_qzeros(inp["qzeros"]).astype(np.float32)
    K, N = q.shape
    g = K // zeros.shape[0]
    s = np.tile(inp["scales"], (g, 1))[:K]  # wrong: tiles groups along K in blocks
    return {"w": ((q - zeros) * s).astype(np.float32)}


# ---------------------------------------------------------------- dequant_gemv (B)

@mutant("dequant_gemv", "gemv_w4a16_gptq", "tile_scales_wrong_broadcast")
def _(inp):
    q = ops.unpack_gptq_qweight(inp["qweight"]).astype(np.float32)
    zeros = ops.unpack_gptq_qzeros(inp["qzeros"]).astype(np.float32)
    K, N = q.shape
    g = K // zeros.shape[0]
    s = np.tile(inp["scales"], (g, 1))[:K]
    w = (q - zeros) * s
    return {"y": (w.T @ inp["x"]).astype(np.float32)}


@mutant("dequant_gemv", "gemv_w4a16_awq", "gptq_decode_of_awq_buffers")
def _(inp):
    """Decodes AWQ-packed buffers with the GPTQ layout."""
    w = ops.dequant_int4_group_gptq(inp["qweight"], inp["qzeros"], inp["scales"])
    return {"y": (w.T @ inp["x"]).astype(np.float32)}


@mutant("dequant_gemv", "gemv_fp8_e4m3", "decode_as_e5m2")
def _(inp):
    w = ops.fp8_to_float(inp["w_bits"], torch.float8_e5m2) * inp["w_scale"][0]
    return {"y": (w.T @ inp["x"]).astype(np.float32)}


@mutant("dequant_gemv", "gemv_w8a8_weight_only", "scale_broadcast_along_K")
def _(inp):
    w = inp["w_q"].astype(np.float32) * inp["w_scales"][:, None]  # wrong axis
    return {"y": (w.T @ inp["x"]).astype(np.float32)}


# ---------------------------------------------------------------- gemm (C)

@mutant("gemm", "gemm_w8a8_dequant", "scales_swapped")
def _(inp):
    acc = inp["x_q"].astype(np.int64) @ inp["w_q"].astype(np.int64)
    y = acc.astype(np.float32) * inp["w_scales"][:, None] * inp["x_scales"][None, :]
    return {"y": y.astype(np.float32)}


@mutant("gemm", "gemm_fp8_e4m3_rowwise", "row_col_scales_swapped")
def _(inp):
    a = ops.fp8_to_float(inp["a_bits"], torch.float8_e4m3fn)
    b = ops.fp8_to_float(inp["b_bits"], torch.float8_e4m3fn)
    y = (a * inp["b_scales"][None, :M(a)]) @ (b * inp["a_scales"][:, None].T)
    return {"y": y.astype(np.float32)}


def M(a):
    return a.shape[0]


@mutant("gemm", "gemm_w4a16_gptq", "group_off_by_one")
def _(inp):
    q = ops.unpack_gptq_qweight(inp["qweight"]).astype(np.float32)
    zeros = ops.unpack_gptq_qzeros(inp["qzeros"]).astype(np.float32)
    K, N = q.shape
    g = K // zeros.shape[0]
    s = np.roll(np.repeat(inp["scales"], g, axis=0), -g, axis=0)  # group index shifted
    w = (q - zeros) * s
    return {"y": (inp["x"] @ w).astype(np.float32)}


# ---------------------------------------------------------------- kv_cache (D)

@mutant("kv_cache", "kv_quant_int8_per_token", "quantize_along_S")
def _(inp):
    k = inp["k"]
    amax = np.abs(k).max(axis=-2, keepdims=True)  # channel-wise, not token-wise
    s = np.where(amax == 0, np.float32(1.0), (amax / 127.0).astype(np.float32))
    q = np.clip(np.rint(k / s), -128, 127)
    return {"q": q.astype(np.int8), "scales": np.broadcast_to(s[..., 0, :], k.shape[:-1]).copy()}


@mutant("kv_cache", "kv_quant_int8_per_channel", "quantize_along_D")
def _(inp):
    k = inp["k"]
    amax = np.abs(k).max(axis=-1, keepdims=True)  # token-wise, not channel-wise
    s = np.where(amax == 0, np.float32(1.0), (amax / 127.0).astype(np.float32))
    q = np.clip(np.rint(k / s), -128, 127)
    return {"q": q.astype(np.int8), "scales": np.broadcast_to(s[..., 0], k.shape[:-1]).copy()}


@mutant("kv_cache", "kv_quant_fp8_e4m3_per_token", "no_zero_row_fallback")
def _(inp):
    k = inp["k"]
    amax = np.abs(k).max(axis=-1, keepdims=True)
    s = (amax / 448.0).astype(np.float32)  # 0 stays 0 -> 0/0 = NaN downstream
    with np.errstate(invalid="ignore", divide="ignore"):
        bits = ops.fp8_bytes(np.nan_to_num(k / s, nan=0.0), torch.float8_e4m3fn)
    return {"bits": bits, "scales": s[..., 0]}


@mutant("kv_cache", "kv_dequant_int8_fp16", "no_fp16_rounding")
def _(inp):
    """Dequantizes in fp32 and skips the fp16 storage rounding the contract requires."""
    out = inp["q"].astype(np.float32) * inp["scales"][..., None].astype(np.float32)
    return {"out": out.astype(np.float32)}


# ---------------------------------------------------------------- fused_longtail (E)

@mutant("fused_longtail", "rmsnorm_fp32", "mean_abs_instead_of_mean_square")
def _(inp):
    x, w = inp["x"], inp["w"]
    ms = np.abs(x).mean(axis=-1, keepdims=True)  # wrong statistic
    return {"y": (x / np.sqrt(ms + 1e-6) * w).astype(np.float32)}


@mutant("fused_longtail", "layernorm_residual_quant_int8", "quantize_residual_instead_of_norm")
def _(inp):
    res_out = (inp["x"] + inp["res"]).astype(np.float32)
    y = res_out  # skips the LayerNorm entirely
    amax = np.abs(y).max(axis=-1, keepdims=True)
    s = np.where(amax == 0, np.float32(1.0), (amax / 127.0).astype(np.float32))
    q = np.clip(np.rint(y / s), -128, 127)
    return {"q": q.astype(np.int8), "s": s[..., 0], "res_out": res_out}


@mutant("fused_longtail", "silu_mul_quant_int8", "silu_on_wrong_operand")
def _(inp):
    a, b = inp["a"], inp["b"]
    p = (b / (1.0 + np.exp(-b)) * a).astype(np.float32)  # silu applied to b
    amax = np.abs(p).max(axis=-1, keepdims=True)
    s = np.where(amax == 0, np.float32(1.0), (amax / 127.0).astype(np.float32))
    q = np.clip(np.rint(p / s), -128, 127)
    return {"q": q.astype(np.int8), "s": s[..., 0]}


@mutant("fused_longtail", "rope_apply", "rotate_half_convention")
def _(inp):
    """Uses the rotate-half (split-D) convention instead of interleaved pairs."""
    q, cos, sin = inp["q"], inp["cos"], inp["sin"]
    D = q.shape[-1]
    h = D // 2
    q1, q2 = q[:, :h], q[:, h:]
    out = np.empty_like(q)
    out[:, :h] = q1 * cos - q2 * sin
    out[:, h:] = q2 * cos + q1 * sin
    return {"out": out.astype(np.float32)}


@mutant("fused_longtail", "moe_router_topk", "ties_by_higher_index")
def _(inp):
    logits, k = inp["logits"], int(inp["k"][0])
    z = logits - logits.max(axis=-1, keepdims=True)
    p = np.exp(z)
    p = (p / p.sum(axis=-1, keepdims=True)).astype(np.float32)
    order = np.argsort(p, axis=-1, kind="stable")[..., ::-1]  # ties: higher index first
    idx = order[..., :k].astype(np.int64)
    w = np.take_along_axis(p, idx, axis=-1)
    w = (w / w.sum(axis=-1, keepdims=True)).astype(np.float32)
    return {"probs": p, "idx": idx, "w_topk": w}
