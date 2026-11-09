# Rounding and scale/zero-point conventions across formats

Companion to `int4_w4a16_packing.md` and `fp8_e4m3_e5m2.md`. Sources: the files cited there
(sha256-pinned), plus torch 2.15.0 behavior verified in the `qkb` env.

## Rounding modes

| Operation | Reference behavior | Common wrong variant |
|---|---|---|
| Float quantization (AWQ/GPTQ): `round((w - zp)/s)` | `torch.round` = **half-to-even** | `floor(x+0.5)` half-up |
| FP8 cast (both formats) | RNE (see fp8 doc, verified halfway cases) | truncation; half-up |
| INT8 symmetric per-channel | RNE via torch.round | half-up; floor |
| FP8→FP16/BF16 upcast | exact (no rounding needed) | — |

Half-up vs half-even agree everywhere except exact halfway values, so detection inputs must
include halfway cases (e.g. scale such that (w-zp)/s = k + 0.5 exactly).

## Scale granularity semantics

- **per-tensor**: one scalar; broadcasting trivial.
- **per-channel (output row)**: scale shape (N, 1) in PyTorch nn.Linear convention (N = out
  features). Bug axis: broadcasting along the wrong axis (K instead of N).
- **per-group along K** (W4A16 standard, group_size = 32/64/128): scales shape
  (N, K/group_size); dequant repeats each scale group_size times along K. Bug axes: off-by-one
  group index at K boundaries; repeat_interleave vs repeat (tile) confusion; wrong group_size
  at the last partial group when K % group_size != 0 (real GPTQ models pad; our task specs must
  state whether K is divisible).

## Zero-point conventions (the single most error-prone axis)

| Format | Convention |
|---|---|
| GPTQ/fbgemm 4-bit | `q = round((w + z*s)/s)`, stored zero `z`; kernels store `z+1` in qzeros (**+1 offset**, reserving 0); dequant `w ≈ (q - z) * s` |
| AWQ | symmetric quantizer; exllama-compatible packing also uses the +1 zero offset (`izeros - 1` on unpack) |
| INT8 symmetric (W8A8) | zero = 128 or 0 with signed int8; must be stated per task |
| FP8 | **no zero point**; scale only (per-tensor, per-row/per-col, or block scale E8M0 for MX formats) |

Bug axes: missing +1 offset; applying +1 on the wrong side (pack vs unpack); treating zero as
signed vs unsigned int4 ([-8,7] vs [0,15]); zero_point dtype cast (float16 rounding of
zero*scale).

## What this file is for

Every "bug axis" named above maps to (a) at least one mutant in `mutants/` (Step 0.7) and
(b) at least one adversarial input class in the hidden test generators (Step 0.6).
