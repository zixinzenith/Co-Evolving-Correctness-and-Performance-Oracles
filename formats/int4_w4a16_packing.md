# INT4 weight-only packing layouts (AWQ vs GPTQ/fbgemm)

Fetched 2026-11-09 via jsDelivr CDN (proxy down at the time; GitHub raw unreachable). jsDelivr
serves the repo's default branch HEAD; each source is cited with both the repo HEAD commit hash
(GitHub API, verified 2026-11-09) and the sha256 of the exact bytes fetched, so the snapshot is
reproducible even if the branch moves.

Sources:
- AutoAWQ, `awq/utils/packing_utils.py` — repo HEAD 88e4c76b20755db275574e6a03c83c84ba3bece5 (main), file sha256 `65eab3eabe3f55e300ffbab5feac59c49322d985f42dcda4e2288859fb9a4abe`
- AutoGPTQ (repo moved to AutoGPTQ/AutoGPTQ), `auto_gptq/nn_modules/qlinear/qlinear_cuda.py` — repo HEAD 9f7d37072917ab3a7545835f23e808294a542153 (main), file sha256 `5a6a483784adc7d9ee55e62f6dc4fdaf3a642f56796d2a6b8afea0b2364d751d`
- GPTQ (reference impl, IST-DASLab/gptq; default branch is now `main`), `quant.py` — repo HEAD 2d65066eeb06a5c9ff5184d8cebdf33662c67faf (main), file sha256 `211528568b962b868d20fc58b55200b615f41ebd9f36b4d2d7e4a196a095ea5a`

## 1. GPTQ / fbgemm layout (the "classic" layout, used by AutoGPTQ cuda kernels)

Quantization: asymmetric per-group. `q = round((w + zero*scale) / scale)`, i.e.
`round((w - zp) / scale)` with `zp = -zero*scale` in signed terms; symmetric option sets
`zero = (maxq+1)/2`. Quantization uses `torch.round` → **half-to-even**.

Packing (from `qlinear_cuda.py::pack`, bits=4):
- `qweight`: shape `(infeatures // 32 * bits, outfeatures)`, int32. The weight is **transposed
  first** (K becomes rows). 8 consecutive int4 values along **K** are packed into one uint32,
  little-endian nibble order: `qweight[row] |= intweight[j] << (4 * (j - i))`, j = i..i+7.
  So the FIRST K element occupies the LOW nibble of the first word.
- `qzeros`: `zeros -= 1` **before packing** (the +1 convention: kernels add 1 back at
  inference, reserving the all-zero code). Shape **(K/g, N/8)** int32 — packed along **N**
  (note the asymmetry: GPTQ packs qweight along K but qzeros along N):
  `qzeros[:, col] |= zeros[:, j] << (4 * (j - i))`, j = i..i+7 iterating over N.
- `scales`: shape `(N, K/g)` (out-features rows, groups along K); exllama variants transpose
  to `(K/g, N)`.
- Exllama/exllamav2 kernels additionally **interleave columns** (see AutoGPTQ
  `qlinear_exllama.py` and `tests/test_repacking.py`); the exllama path re-packs rather than
  re-quantizing.

## 2. AWQ layout

**Empirically verified 2026-11-12** by running AutoAWQ's own `unpack_awq` + `reverse_awq_order`
end-to-end round-trips in the `qkb` env (see `tasks/common/` tests); buffer shapes cross-checked
against `awq/modules/linear/gemm.py::WQLinear_GEMM.__init__`. An earlier version of this section
described the orientation incorrectly — this version is the verified one.

- Buffer shapes (in_features = K, out_features = N, group_size = g, 4-bit):
  - `qweight`: **(K, N/8)** int32 — packs along **N**, not K! (Transposed storage relative to
    GPTQ's qweight.)
  - `qzeros`: **(K/g, N/8)** int32 — also packs along N.
  - `scales`: **(K/g, N)** fp16.
- Nibble order: nibble j of word (k, c) of qweight holds the logical value at **N-position
  8c + AWQ_ORDER[j]**, where `AWQ_ORDER = [0, 2, 4, 6, 1, 3, 5, 7]` — i.e. even logical columns
  occupy the low nibbles, odd columns the high nibbles, within each 8-column block. The same
  rule applies to qzeros. `AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]` is its inverse.
- Zero-point: the AWQ **checkpoint format stores zeros raw (no −1/+1 offset)**. The +1 offset
  appears only in `unpack_reorder_pack`, which converts to the exllama kernel format ("exllama
  adds 1 during inference"). Dequant from checkpoint format is simply
  `w[k, n] = (q[k, n] − z[k/g, n]) · s[k/g, n]` (verified: `dequantize_gemm` does no offset).
- Quantization: AWQ's quantizer is symmetric with a grid-searched scale; see
  `awq/quantize/quantizer.py`.

Packing pitfall for implementers: nibble 7 with value ≥ 8 overflows int32 if packed in int32
arithmetic — pack in int64 (or uint32) and cast down, and mask with 0xF after unpacking
(AutoAWQ itself truncates to int8 mid-unpack and masks only afterwards).

## 3. Bit-exactness rules for the verifier

- pack/unpack and layout-conversion tasks: **byte-exact** comparison against a NumPy reference
  (integer domain, no tolerance).
- The nibble order, the interleaving permutation, the packing axis (K for qweight, K-columnwise
  for qzeros), and the ±1 zero-point offset are the four independent bug axes; each needs at
  least one mutant in Step 0.7.
- A kernel that unpacks with the wrong convention can still pass a loose float tolerance on
  small random weights (the errors average out) — this is exactly why KernelBenchX-style checks
  cannot catch layout bugs.
