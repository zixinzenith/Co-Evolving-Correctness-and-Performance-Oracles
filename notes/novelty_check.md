# Step 0.1 — Novelty Check

Date: 2026-11-03
Method: arXiv API (via proxy), GitHub API, full-text read of the project's own survey (`notes/survey/AI_Automatic_Operator_Generation_Literature_Review.docx`).

## Verdict

**(b) Adjacent work exists — no direct competitor found. The gap is confirmed, with one important nuance.**

No dedicated benchmark or study that specifically and substantially evaluates LLM/agent generation of *quantized* kernels exists as of 2026-11-03. KernelBenchX is the only benchmark with a quantization category, and it treats quantization as one of 15 categories in a general kernel benchmark — 6 task files, reported only as an aggregate "0/30". There is no quantization-specific verification methodology, no error taxonomy, no comparison against expert quantized-kernel libraries, and no intervention study. Our proposed contributions (bit-exact verification where the operation is exactly defined, format-derived tolerances elsewhere, expert baselines like Marlin/torchao/bitsandbytes, RQ2 error taxonomy, RQ3 interventions) are all unclaimed territory.

**The nuance:** because KernelBenchX's quantization category is small (6 tasks) and its checks are loose, our RQ1 must be positioned as "first systematic study with quantization-appropriate verification", not merely "first benchmark with quantization tasks".

## Verification of the survey's cited numbers

Both load-bearing numbers in the survey check out against the arXiv abstracts:

1. **KernelBenchX, 0/30 on quantization** — VERIFIED.
   arXiv:2605.04956 ("KernelBenchX: A Comprehensive Benchmark for Evaluating LLM-Generated GPU Kernels", 2026-05, v2 2026-05-11). Abstract states verbatim: "quantization remains completely unsolved (0/30 successes) despite non-trivial compilation rates, revealing systematic misunderstanding of numerical computation contracts rather than surface-level syntax errors."
   Repo: https://github.com/BonnieW05/KernelBenchX (active, last push 2026-10-23).
   The 30 = 6 quantization task files × 5 evaluated methods. Task files in `data/kernelbenchx/Quantization/`: attention_w8a8, bmm_w8a8, conv2d_w8a8, layernorm_w8a8, linear_w4a16, matmul_w8a8.

2. **78.6% precision faults missed by the official checker** — VERIFIED.
   arXiv:2609.22220 ("Measuring the Checker: Mutation Analysis for GPU-Kernel Benchmark Oracles", 2026-09). Abstract states verbatim: "8.7% of arithmetic faults escape, but 78.6% of precision faults do." Artifact: KernelBench-M on HuggingFace.

## How KernelBenchX handles quantization (read from source) — differentiation for us

- Their quantization tasks are *runtime dynamic quantization described in a docstring* (e.g. `linear_w4a16.py` specifies per-output-channel symmetric int4, scale = max(abs(w))/7, clamp [-8,7], low-nibble-first packing, dequant-on-the-fly). The reference "implementation" is just `F.linear` — i.e. there is no bit-exact reference at all.
- Their accuracy requirement is **very loose**: cosine_sim >= 0.90, l1_relative <= 0.10, rmse <= 0.15. A loose allclose-style check hides exactly the bugs we care about (bit order, scale broadcast, rounding mode, overflow).
- Their `utils/quantization_checker.py` self-describes as a "weak filter to exclude trivial non-quantized implementations, **not a full semantic verifier**" — it is AST/regex pattern matching for scale-derivation idioms, plus forbidden-API lists.
- Conclusion: KernelBenchX cannot tell *why* quantization generation fails, and its loose tolerances may both pass wrong kernels and fail right ones. Our verifier design is the core delta.

## Adjacent work found (not competitors)

| Work | What it is | Why not a competitor |
|---|---|---|
| KernelBenchX (arXiv:2605.04956) | 176-task LLM kernel benchmark, 15 categories incl. quantization (6 tasks) | Quantization is one category, loose checks, no quantization-specific analysis or expert baselines |
| AKG kernel Agent (arXiv:2512.23424) | Multi-agent kernel synthesis framework (Triton/TileLang/CPP/CUDA-C, GPU+NPU) | Evaluated on KernelBench FP32 ops; quantization appears only as motivation |
| scratchbench (github.com/selimfedakar/scratchbench) | 17-task laptop benchmark: implement ML machinery (attention masks, KV-cache, one quantization round-trip, one Triton kernel) from spec, graded by hidden pytest | Single-author, 0 stars; correctness-only (no performance vs expert kernels), one quantization task, no error taxonomy |
| Accelerating a Triton Fused Kernel for W4A16 (arXiv:2402.00025) | Hand-written fused dequant-GEMM SplitK kernel (A100/H100) | Human-written; useful as a technical reference, not agent-focused |
| W4A16 on Ascend NPU (arXiv, 2026), QServe, LiquidGEMM, Marlin, torchao, bitsandbytes, CUTLASS mixed-input, torch._scaled_mm | Expert quantized-kernel libraries / designs | Baselines for us, not evaluations of LLM generation |

Searches that found nothing relevant (arXiv full-text, all-fields, relevance-sorted): "quantized kernels" AND "LLM"; "quantization" AND "kernel generation"; "W4A16"; "quantized operator"; "FP8" AND "kernel generation". GitHub repo search: "LLM quantized kernel benchmark", "quantization kernel generation LLM" — no dedicated benchmark repos. (Caveat: GitHub search only sees repo names/descriptions/readmes, not deep code; a quantization category buried inside a general agent-kernel repo could be missed. Nothing in the survey or citations points to one.)

## Unverified / caveats

- The per-task results table inside KernelBenchX's PDF (which specific models fail which of the 6 tasks) was not read — only the abstract and repo source. Worth reading in Step 0.2.
- Humming (vLLM, Oct 2026) was not independently re-verified in this step; it is in the survey and will be checked in Step 0.2.
- 0/30 interpretation as 6 tasks × 5 methods is inferred from repo structure ("30 successes" denominator); the paper's table should confirm in 0.2.
