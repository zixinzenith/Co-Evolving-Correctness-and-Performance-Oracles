# tasks/

Task definitions: ~40 quantized operators in 5 groups.

1. quantize / pack-unpack
2. fused dequant-GEMV
3. W4A16 / W8A8 GEMM
4. KV-cache quantization
5. fused long-tail operators

Each task will carry a reference implementation, input generators, and a pointer to its format spec in `formats/`. Covers both runtime dynamic quantization and pre-quantized packed-weight layouts (AWQ/GPTQ).
