# mutants/

Hand-written faulty variants of reference kernels — one per documented bug axis from
`formats/rounding_and_scale_conventions.md`: wrong nibble order, missing zero-point offset,
half-up rounding, scale broadcast on the wrong axis, wrong quantization axis, fp8 format
confusion, dropped scale fallbacks, wrong reduction semantics, tie-breaking order.

12 mutants cover the `quant_pack` group (the bit-exact oracle core), 3-5 per group for the
rest — 28 total, registered in `variants.py` as `MUTANTS[(group, task)] -> [(name, fn)]`.

`run_mutants.py` feeds every mutant to the verifier over all shapes × seeds and reports the
detection rate — the oracle-adequacy measure (cf. arXiv:2609.22220, where the official
KernelBench checker misses 78.6% of precision faults). Current suite: **570/570 detected**.
An escape would mean the format-derived tolerance is too loose for that fault class and the
bound needs tightening — that feedback loop is the point of the exercise.

```bash
make mutants
```
