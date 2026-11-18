# Task spec schema

Each task lives in `tasks/<group>/<name>/` with two files:

- `spec.yaml` — the declarative spec. This is what gets rendered into the agent's prompt
  (minus the `hidden` shapes).
- `impl.py` — `ref(**inputs) -> outputs` (ground truth) and `gen_inputs(shape, seed) -> dict`
  (adversarial input generator, deterministic given (shape, seed)).

## spec.yaml fields

| Field | Meaning |
|---|---|
| `name`, `group` | Task id and group (`quant_pack`, `dequant_gemv`, `gemm`, `kv_cache`, `fused_longtail`) |
| `description` | 2-3 sentences for the agent prompt, including what makes the task hard |
| `inputs` / `outputs` | Semicolon list: `name dtype (shape_expr)`; shape exprs use symbols K, N, M, g with `/` division, e.g. `K/8`, `K/g`, `M,1` |
| `constraints` | Divisibility constraints the generator guarantees |
| `format_ref` | The formats/*.md section defining the bit-level contract |
| `verification.mode` | `bit_exact` (byte-exact vs ref) or `derived_tolerance` (tolerance formula stated in spec) |
| `verification.tolerance` | For `derived_tolerance`: `kind: sum_abs` with `coef` (× the format error scale) or `kind: elementwise` with `rtol`/`atol`; `int_shift` = codes a float-math reduction may flip (0 = exact) |
| `shapes.visible` / `shapes.hidden` | Concrete shape dicts. Visible shapes go in the prompt; hidden are eval-only and include boundary stress (non-power-of-two, extreme sizes) |
| `baseline` | `pytorch` equivalent, `torch_compile` applicability, `expert` kernel (marlin/humming/torchao/bitsandbytes/_scaled_mm/null) — Phase 1 only |
| `adversarial_classes` | Which adversarial input classes the generator exercises (see formats/rounding_and_scale_conventions.md) |

## Ground-truth rules

- Group A (quant_pack): integer-domain numpy references; verification is byte-exact.
- GEMM/GEMV groups: fp32/fp64 references; tolerance derived per-task from the format error
  bound, never `torch.allclose` defaults. Derived-tolerance tasks implement
  `tol_scale(**inputs)` in `impl.py` — the format's error scale (e.g. half a code step per
  weight weighted by |x|, or the fp8 mantissa error); the verifier bounds candidates by
  `coef * tol_scale + atol` (or elementwise `rtol`/`atol`), so the bound is stated in the
  spec and enforced mechanically.
- Layout contracts come from formats/*.md, which are pinned to upstream sources by commit
  hash + file sha256.

## Validation

`make tasks-validate` (harness/validate_tasks.py) checks, for every task and every visible
+ hidden shape, seeds 0-2: schema completeness, gen/ref key agreement with spec, output
sizes/dtypes, byte-level determinism, and dtype sanity for the declared verification mode.
It regenerates `tasks/manifest.json`.
