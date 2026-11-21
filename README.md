# QuantKernelBench

**Can LLM agents write correct and fast quantized operators?**

Nearly all LLM inference today runs on quantized formats — W4A16, W8A8, FP8, NVFP4 — yet the GPU kernels that implement these formats are still hand-written by a small set of experts. Research on LLM kernel generation has focused almost exclusively on FP32/FP16 operators. KernelBenchX (arXiv:2605.04956) shows all five evaluated methods scoring **0/30 on quantization tasks** despite non-trivial compile rates: the failures stem from models not treating the *numerical contract* (packing layout, rounding mode, scale broadcast, saturation) as part of the computation, not from surface syntax. The area has been called a "clear gap"; no systematic study has followed up.

## What this is

A benchmark of **40 quantized operators** in five groups:

| Group | Tasks | Verification |
|---|---|---|
| A. quantize / pack-unpack | 12 | bit-exact |
| B. fused dequant-GEMV | 8 | derived tolerance |
| C. W4A16 / W8A8 GEMM | 6 | bit-exact (int32 accum) + derived |
| D. KV-cache quantization | 7 | bit-exact (int codes / fp8 bytes) |
| E. fused long-tail operators | 7 | derived tolerance |

It covers both runtime dynamic quantization and the pre-quantized packed-weight layouts (AWQ/GPTQ) that production inference actually uses. Three things distinguish it from existing kernel benchmarks:

1. **Quantization-appropriate verification.** Bit-exact checks for operations with an exact definition (pack/unpack, casts with a specified rounding mode); tolerances for GEMM/GEMV derived from the format's error bound and stated in each task spec — never a loose `allclose` (published checkers miss 78.6% of precision-class faults, arXiv:2609.22220). The oracles themselves are mutation-tested (`mutants/`).
2. **Expert-aligned baselines.** PyTorch eager, `torch.compile`, and the expert quantized-kernel stack: Marlin, vLLM Humming, torchao, bitsandbytes, `torch._scaled_mm` (Phase 1).
3. **Error taxonomy and interventions.** RQ1: per-model capability on quantized kernels. RQ2: a fault taxonomy for quantization kernels, grounded in the adversarial input classes of `formats/rounding_and_scale_conventions.md`. RQ3: cheap interventions — format spec cards, bit-exact unit-test feedback, retrieved expert examples — via the four prompt conditions in `prompts/`.

No model training. Phase 0 (task set + verifier, CPU-only) is complete; Phase 1 runs the performance evaluation on a rented Ada GPU (RTX 4090 / L40S, SM89). NVFP4/DeepGEMM tasks are deferred to an optional Hopper/Blackwell stage.

## Layout

| Path | Contents |
|---|---|
| `notes/` | literature survey, novelty check, related-work review (md + docx) |
| `formats/` | quantization format specs — AWQ/GPTQ packing, FP8 e4m3/e5m2, rounding & zero-point conventions; every source pinned by commit hash + sha256 |
| `tasks/` | the 40 task specs (`spec.yaml`) + reference implementations (`impl.py`) + adversarial generators |
| `verifier/` | bit-exact checking, format-derived tolerances, anti-cheating checks |
| `mutants/` | hand-written faulty variants + detection-rate script (oracle adequacy) |
| `agent/` | the generation loop (fixed 6 iterations, identical across models) |
| `prompts/` | the 4 prompt conditions |
| `harness/` | task validator and run driver (JSONL records in `runs/`) |
| `BUDGET.md`, `STATUS.md` | cost model and progress |

## Quickstart

```bash
make setup     # create the pinned CPU environment (python 3.11, torch 2.15 CPU, numpy 2.4)
make test      # validate all task specs against the schema + run the test suite
make mutants   # run the mutant suite against the verifier, print detection rates
```

Single-task end-to-end (fake model, no API key needed):

```bash
make run-e2e
```

## Status

Phase 0 complete: 40/40 tasks validated (`tasks/manifest.json`), verifier + mutation analysis in place, agent loop e2e-tested. See `STATUS.md` for the checklist and `BUDGET.md` for the Phase 1 cost model.

## License

MIT
