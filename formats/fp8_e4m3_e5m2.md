# FP8 formats: e4m3fn and e5m2

Behavior below is **empirically verified** in our pinned environment (torch 2.15.0 CPU, macOS
arm64, `qkb` env) on 2026-11-11 — see `verifier/tests/` for the permanent regression test.
GPU cast/cuBLASLt `_scaled_mm` behavior is re-verified on the Phase 1 instance; flag any
divergence there.

Reference: OCP FP8 specification (e4m3 = 1-4-3 "fn" finite variant, e5m2 = 1-5-2 IEEE-like).

## Bit layout and limits (torch.finfo, verified)

| Property | e4m3fn | e5m2 |
|---|---|---|
| Sign / exponent / mantissa bits | 1 / 4 / 3 | 1 / 5 / 2 |
| Max finite | 448.0 | 57344.0 |
| Min normal (tiny) | 2^-6 = 0.015625 | 2^-14 = 6.103515625e-05 |
| Min subnormal | 2^-9 = 0.001953125 | 2^-16 |
| Inf representable | **No** ("fn": the inf bit pattern is a NaN) | Yes |
| NaN | Yes (0x7F/0xFF and S.1111.111) | Yes |

## Cast behavior on overflow (verified in torch 2.15.0)

```
e4m3fn: [448, 469, 1e6, inf, nan].to(float8_e4m3fn) -> [448, 448, 448, 448, nan]
e5m2:   [57344, 60000, 1e6, inf, nan].to(float8_e5m2) -> [57344, 57344, inf, inf, nan]
```

- **e4m3fn saturates on cast**: everything above 448 (including float inf) becomes 448, not inf.
  This is the torch CPU cast behavior and the single most common FP8 bug source (naive
  implementations produce inf or NaN where the reference saturates).
- **e5m2 preserves inf**: values modestly above max (60000) round down to 57344; large values
  (1e6) and inf become inf.

## Rounding (verified)

Casts use **round-to-nearest-even** (RNE). Verified halfway cases:
- e4m3fn: 1.0625 → 1.0 (ties to even mantissa 000), 1.1875 → 1.25 (mantissa 010).
- e5m2: 1.25 and 1.75 are exactly representable (2 mantissa bits).
- Subnormal halfway: 2^-10 → 0 (ties to even).
- e4m3fn min subnormal 2^-9 is exactly representable.

Note: `torch.round` on floats is also half-to-even; AWQ/GPTQ integer quantization therefore
inherits RNE. A half-up (round-half-away) implementation differs exactly on halfway values —
mutant class "wrong rounding mode" must test these.

## Verifier rules

1. Pure casts / dtype conversions / pack-unpack of FP8: **bit-exact** (compare uint8 views).
2. FP8 GEMM: tolerance derived from the error bound. With per-tensor scale s, both operands
   e4m3fn, accumulation in fp32: relative tolerance driven by 2^-3.5 mantissa error per operand
   (approx 6% worst-case per element, sqrt-accumulated). Derivation written into the task spec,
   not chosen ad hoc.
3. Saturation tests are mandatory inputs: include 448-boundary, just-above-max, inf, NaN
   patterns, and the subnormal minimum.
