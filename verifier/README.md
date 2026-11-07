# verifier/

Correctness checking for generated kernels.

- **Bit-exact** checks for exactly defined operations (pack/unpack, quantize with a specified rounding mode).
- **Derived tolerances** for GEMM/GEMV-style operations: bounds computed from the format's error model against a float64 reference, not a loose `allclose`.
- **Anti-cheating checks** (e.g. PyTorch fallback detection, hidden inputs).

`tests/` holds the verifier's own unit tests (currently an environment smoke test).
