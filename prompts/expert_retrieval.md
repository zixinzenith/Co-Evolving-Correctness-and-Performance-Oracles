# Condition 4 — + retrieved expert kernel examples

The baseline prompt plus a pointer to (Phase 1: the relevant excerpt of) the expert kernel
the task's baseline is drawn from — Marlin's global→shared dequant path for W4A16 GEMM,
vLLM Humming's WNA16 strategy, torch._scaled_mm scale semantics for FP8, and so on.

Tests whether grounding in working expert code beats grounding in prose (compare with
condition 2).
