# Step 0.2 — Related Work Review

Date: 2026-11-05
Sources: arXiv API + paper HTML (via proxy), GitHub API + raw READMEs. All abstract quotes verbatim from arXiv.

## A. KernelBenchX in depth (arXiv:2605.04956, repo BonnieW05/KernelBenchX)

The single most important prior work for us. 176 tasks, 15 categories, five methods: **AutoTriton, GEAK, KernelAgent, Claude, KernelLLM (DeepSeek-Coder)**.

Quantization category (6 tasks: matmul_w8a8, bmm_w8a8, conv2d_w8a8, layernorm_w8a8, attention_w8a8, linear_w4a16):

| Metric | AutoTriton | GEAK | KernelAgent | Claude | KernelLLM |
|---|---|---|---|---|---|
| Correctness | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Compile rate | 0.0 | 50.0 | 83.3 | 33.3 | 0.0 |

Overall correctness across all categories: GEAK 30.7%, Claude 22.7%, AutoTriton 17.0%, KernelAgent 10.8%, KernelLLM 0%.

Key takeaways for our design:
- **Compile ≠ correct**: KernelAgent compiled 83.3% of quantization tasks and got 0 correct. The paper attributes this to "systematic misunderstanding of numerical computation contracts rather than surface-level syntax errors" — models "do not reliably treat the numerical contract itself as part of the computation to be preserved". This is precisely our RQ2.
- Their quantization verification uses custom loose thresholds (e.g. matmul_w8a8: cosine ≥ 0.95, l1_rel ≤ 0.05, rmse ≤ 0.10) plus a static "quantization evidence" regex/AST checker that self-describes as not a semantic verifier. No bit-exact checks, no packing-layout verification, no expert-kernel baselines.
- Their 6 tasks are runtime dynamic quantization (quantize inside the kernel at runtime), not the packed-weights regime (AWQ/GPTQ pre-quantized weight layouts) that production inference actually uses. Our task set covers both, including pack/unpack and pre-quantized GEMM/GEMV — no overlap concern.

## B. Benchmark / verification papers (agent-adjacent)

| Paper | arXiv | One-line relevance to us |
|---|---|---|
| KernelBenchX | 2605.04956 | Quantization 0/30; motivation for RQ1/RQ2 (see §A) |
| Measuring the Checker (KernelBench-M) | 2609.22220 | Official KernelBench checker misses 1/6 faults; **78.6% of precision faults escape**; mutation analysis as oracle-adequacy metric — we adopt mutation testing (Step 0.7) for the same purpose |
| KernelBench-Verified | 2607.16241 | Headline speedup falls 1.43× → 0.88× under honest protocol (TF32 baseline, hidden inputs, memory metrics); "no model consistently outperforms PyTorch" — motivates our honest-baseline policy and anti-cheating checks |
| KernelMutate | 2611.15441 | Mutation-guided validation of generated kernels; five-draw anchor accepts 548/838 CUDA candidates, then faults surface in FP32/dtype witnesses — more evidence that default differential testing is too weak for dtype-sensitive kernels |
| AKG kernel Agent | 2512.23424 | Multi-agent kernel synthesis (Triton/TileLang/CPP/CUDA-C, GPU+NPU), 1.46× on KernelBench — system to potentially include as a method, but it has no quantization evaluation |
| Atrex-Bench | 2607.14541 | Production-trace operators; best model ~10% of roofline; much of pass rate is PyTorch fallbacks — motivates our runtime fallback detection |

## C. Expert quantized-kernel libraries (our baselines) — GPU architecture support

| Library / kernel | Formats | Arch support | Role in QuantKernelBench |
|---|---|---|---|
| **vLLM Humming** (vllm-project/humming, JIT GEMM library) | Weights: symmetric INT1–8, INT w/ dynamic zero-point, arbitrary signed FP (kBits ≤ 8); strategies WNA16, W8A8, W4A8, W4A4, MXFP4/MXFP8, NVFP4; scales BF16/FP16/E4M3/E5M2/UE8M0 | **SM75 (Turing) onward**, incl. Hopper, Blackwell (SM100/103/110/120/121), Rubin; also Zhenwu 810E. FP8 needs SM89+, FP4 needs SM100+ | The strongest modern baseline; README claims ~2–5× over vLLM Marlin / DeepGEMM on H200/B300 |
| **Marlin** (IST-DASLab/marlin) | FP16×INT4 weight-only, batch ≤ 16–32 sweet spot | SM80+ (Ampere; tuned for A100/H100) — confirm at Phase 1 | Classic W4A16 expert baseline |
| **torchao** (pytorch/ao) | Int4 weight-only, FP8 rowwise, MXFP8, NVFP4 (via MSLK kernels) | Follows PyTorch; FP8/NVFP4 kernels need SM89+/SM100+ | PyTorch-native baseline; also source of reference quantization schemes |
| **bitsandbytes** | LLM.int8() (vector-wise w/ outlier handling), multi-vendor (NVIDIA/AMD/Intel) | Broad; CUDA/XPU | W8A8-style baseline |
| **CUTLASS mixed-input GEMM** | INT4/INT8 weights × FP16/BF16 activations | SM80+ (Ampere collective builder); Hopper SM90 variants — re-verify exact example SM targets at Phase 1 | Template reference for expert kernel structure |
| **torch._scaled_mm** | FP8 e4m3/e5m2 GEMM w/ scales, cuBLASLt backend | SM89+ (rowwise); some paths SM90 — re-verify at Phase 1 | FP8 baseline that our generated FP8 GEMMs must compete with |
| **QServe (QoQ)** | 2405.04532 | W4A8KV4 co-design; diagnoses 20–90% dequant overhead in naive INT4 kernels — good motivation for fused dequant-GEMV tasks | Related system, W4A8 baseline lineage |
| **LiquidGEMM** | 2509.01229 | W4A8 with overflow-safe dequant (LiquidQuant); targets SM80/SM89/SM90 (re-verify) | W4A8 expert baseline |
| DeepGEMM | FP8 blockwise | Hopper-only (SM90) | Optional H100-phase baseline only |
| **scratchbench** (selimfedakar/scratchbench) | 17 laptop tasks incl. one quantization round-trip, hidden pytest | CPU | Adjacent small benchmark; cite in related work |

## D. Hardware implication for Phase 1

Survey's earlier note holds and is now grounded: an **Ada GPU (RTX 4090 / L40S, SM89)** covers INT4/INT8, Marlin, Humming's FP8 paths, and torch._scaled_mm rowwise FP8 — that is enough for the W4A16/W8A8/FP8 core of the task set. NVFP4/MXFP4 tasks (SM100+) and DeepGEMM (SM90) move to the optional Hopper/Blackwell phase, or are verified numerically on CPU with their kernels marked "requires SM100, not benchmarked here".

## E. Open items carried to Phase 1

- Exact SM targets of CUTLASS mixed-input examples and torch._scaled_mm rowwise/tensorwise paths — re-verify against docs/hardware when instances exist.
- Humming's benchmark tables (H200/B300) are vendor-reported; our runs will produce our own numbers on our hardware.
