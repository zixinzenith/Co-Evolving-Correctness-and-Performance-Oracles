# verifier/

Correctness checking for generated kernels.

- `check.py` — the oracle. Two modes, per `spec.yaml`:
  - **bit-exact**: byte-identical outputs for exactly defined operations (pack/unpack,
    integer quantization with a specified rounding mode, fp8 casts compared as uint8).
  - **derived tolerances**: integer outputs exact (or within `int_shift` codes where float
    reduction order may flip a halfway code); float outputs bounded by the format's error
    model — `sum_abs` (coef × format error scale, computed by the task's `tol_scale()`) or
    `elementwise` (rtol/atol). Never loose `allclose` defaults.
- `anticheat.py` — forbidden-import scan (the candidate may not import the oracle stack),
  output-aliasing detection, and hidden-shape evaluation as the structural anti-cheat.
- `tests/` — the verifier's own unit tests: the oracle is the load-bearing component, so it
  is tested directly (a wrong oracle would silently accept wrong kernels).
