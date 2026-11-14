"""One-off generator for Group A (quant_pack) task files. Kept for provenance.

Run from repo root: python scripts/gen_group_a.py
Emits tasks/quant_pack/<name>/spec.yaml and impl.py for each task in the table below.
"""

import os
import textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VISIBLE = [
    {"K": 64, "N": 16, "g": 32, "M": 4},
    {"K": 128, "N": 32, "g": 64, "M": 8},
]
HIDDEN = [
    {"K": 256, "N": 64, "g": 128, "M": 16},
    {"K": 512, "N": 128, "g": 64, "M": 1},
    {"K": 1024, "N": 256, "g": 128, "M": 32},
    {"K": 96, "N": 24, "g": 48, "M": 2},   # non-power-of-two stress
    {"K": 4096, "N": 512, "g": 128, "M": 64},
]

# name, description, inputs, outputs, ref-body, gen-body
TASKS = [
    (
        "int4_quantize_asym",
        "Asymmetric per-group INT4 quantization with round-half-to-even. "
        "The +1 wrap of the zero point and the rounding mode are the bug axes.",
        "w fp32 (K,N); scales fp32 (K/g,N); zeros fp32 (K/g,N)",
        "q int64 (K,N) in [0,15]",
        "return {'q': ops.quantize_int4_asym(inp['w'], inp['scales'], inp['zeros'])}",
        """w = gen.fp16_weights((K, N), seed)
        scales = np.full((K // g, N), 0.0625, dtype=np.float32)
        zeros = np.tile(np.arange(8, dtype=np.float32), (K // g, N // 8))
        if seed % 3 == 1:  # halfway class: (w/s + z) lands exactly on k + 0.5 in fp32
            k = np.arange(K * N, dtype=np.float32).reshape(K, N) % 7
            w = ((k + 0.5) - 8.0) * 0.0625
        return {'w': w, 'scales': scales, 'zeros': zeros}""",
    ),
    (
        "int4_pack_gptq",
        "Pack INT4 codes into the GPTQ/fbgemm qweight layout: (K/8, N) int32, "
        "8 codes along K per word, low nibble first.",
        "q int64 (K,N) in [0,15]",
        "qweight int32 (K/8,N)",
        "return {'qweight': ops.pack_gptq_qweight(inp['q'])}",
        """q = gen.int4_boundary((K, N), seed) if seed % 2 else gen.int4_uniform((K, N), seed)
        return {'q': q}""",
    ),
    (
        "int4_pack_zeros_gptq",
        "Pack zero-points into GPTQ qzeros with the -1 offset wrapping mod 16 "
        "(logical 0 becomes stored nibble 15). Packs along N, unlike qweight.",
        "zeros int64 (K/g,N) in [0,15]",
        "qzeros int32 (K/g,N/8)",
        "return {'qzeros': ops.pack_gptq_qzeros(inp['zeros'])}",
        """z = gen.int4_boundary((K // g, N), seed)
        return {'zeros': z}""",
    ),
    (
        "int4_pack_awq",
        "Pack codes AND zeros into the AWQ layout: qweight (K, N/8), qzeros (K/g, N/8), "
        "both packing along N with the AWQ_ORDER [0,2,4,6,1,3,5,7] interleave; no zero offset.",
        "q int64 (K,N); zeros int64 (K/g,N)",
        "qweight int32 (K,N/8); qzeros int32 (K/g,N/8)",
        """return {'qweight': ops.pack_awq(inp['q']),
                'qzeros': ops.pack_awq(inp['zeros'])}""",
        """q = gen.int4_boundary((K, N), seed) if seed % 2 else gen.int4_uniform((K, N), seed)
        z = gen.int4_uniform((K // g, N), seed + 1)
        return {'q': q, 'zeros': z}""",
    ),
    (
        "int4_unpack_gptq",
        "Unpack GPTQ qweight back to INT4 codes (low nibble = first K position).",
        "qweight int32 (K/8,N)",
        "q int64 (K,N)",
        "return {'q': ops.unpack_gptq_qweight(inp['qweight'])}",
        """q = gen.int4_uniform((K, N), seed)
        return {'qweight': ops.pack_gptq_qweight(q)}""",
    ),
    (
        "int4_unpack_awq",
        "Unpack AWQ qweight/qzeros (AWQ_ORDER interleave along N, no zero offset).",
        "qweight int32 (K,N/8); qzeros int32 (K/g,N/8)",
        "q int64 (K,N); zeros int64 (K/g,N)",
        """return {'q': ops.unpack_awq(inp['qweight']),
                'zeros': ops.unpack_awq(inp['qzeros'])}""",
        """q = gen.int4_boundary((K, N), seed)
        z = gen.int4_uniform((K // g, N), seed + 1)
        return {'qweight': ops.pack_awq(q), 'qzeros': ops.pack_awq(z)}""",
    ),
    (
        "int4_convert_gptq_to_awq",
        "Repack a GPTQ-packed layer into the AWQ layout. Combines both packing conventions "
        "and the differing zero-point offsets (GPTQ -1 wrap removed, AWQ raw).",
        "qweight int32 (K/8,N); qzeros int32 (K/g,N/8)",
        "qweight int32 (K,N/8); qzeros int32 (K/g,N/8)",
        """q = ops.unpack_gptq_qweight(inp['qweight'])
        z = ops.unpack_gptq_qzeros(inp['qzeros'])
        return {'qweight': ops.pack_awq(q), 'qzeros': ops.pack_awq(z)}""",
        """q = gen.int4_uniform((K, N), seed)
        z = gen.int4_boundary((K // g, N), seed + 1)
        return {'qweight': ops.pack_gptq_qweight(q), 'qzeros': ops.pack_gptq_qzeros(z)}""",
    ),
    (
        "int8_quant_per_channel",
        "Symmetric INT8 quantization per output channel: scale = amax(|w|)/127, "
        "round-half-even, clip [-128,127]. Zero channel forces the scale=1 fallback.",
        "w fp32 (K,N)",
        "q int8 (K,N); scales fp32 (N,)",
        """q, s = ops.quantize_int8_per_channel(inp['w'])
        return {'q': q, 'scales': s}""",
        """w = gen.int8_adversarial((K, N), seed)
        return {'w': w}""",
    ),
    (
        "int8_quant_per_token",
        "Dynamic symmetric INT8 quantization per token (row): scale = amax(|x|)/127.",
        "x fp32 (M,K)",
        "q int8 (M,K); scales fp32 (M,1)",
        """q, s = ops.quantize_int8_per_token(inp['x'])
        return {'q': q, 'scales': s}""",
        """x = gen.int8_adversarial((M, K), seed)
        return {'x': x}""",
    ),
    (
        "fp8_cast_e4m3",
        "Cast fp32 to FP8 e4m3fn and return the raw bit patterns. Saturation: everything "
        "above 448 (including inf) becomes 448; NaN stays NaN; rounding is RNE.",
        "x fp32 (M,N)",
        "bits uint8 (M,N)",
        "return {'bits': ops.fp8_bytes(inp['x'], torch.float8_e4m3fn)}",
        """x = gen.fp8_adversarial((M, N), seed, 'e4m3')
        return {'x': x}""",
    ),
    (
        "fp8_cast_e5m2",
        "Cast fp32 to FP8 e5m2 and return the raw bit patterns. Preserves inf; "
        "large values overflow to inf; rounding is RNE.",
        "x fp32 (M,N)",
        "bits uint8 (M,N)",
        "return {'bits': ops.fp8_bytes(inp['x'], torch.float8_e5m2)}",
        """x = gen.fp8_adversarial((M, N), seed, 'e5m2')
        return {'x': x}""",
    ),
    (
        "w4a16_dequant_gptq",
        "Dequantize a full GPTQ W4A16 layer to fp32 weights: unpack codes and zeros "
        "(offset unwrapped mod 16), apply per-group scales. Shape/orientation of all "
        "three buffers is the bug axis.",
        "qweight int32 (K/8,N); qzeros int32 (K/g,N/8); scales fp32 (K/g,N)",
        "w fp32 (K,N)",
        """w = ops.dequant_int4_group_gptq(inp['qweight'], inp['qzeros'], inp['scales'])
        return {'w': w}""",
        """q = gen.int4_uniform((K, N), seed)
        z = gen.int4_boundary((K // g, N), seed + 1)
        s = np.abs(gen.fp16_weights((K // g, N), seed + 2)) * 0.01 + 1e-4
        return {'qweight': ops.pack_gptq_qweight(q),
                'qzeros': ops.pack_gptq_qzeros(z),
                'scales': s.astype(np.float32)}""",
    ),
]

YAML_TMPL = """name: {name}
group: quant_pack
description: |
  {desc}
inputs: {inputs}
outputs: {outputs}
constraints: "K % 8 == 0, N % 8 == 0, K % g == 0"
format_ref: formats/int4_w4a16_packing.md
verification:
  mode: bit_exact
shapes:
  visible: {visible}
  hidden: {hidden}
baseline:
  pytorch: none
  torch_compile: false
  expert: null
adversarial_classes: [boundary_nibbles, zero_offset_wrap, halfway_rounding, saturation, outliers, zero_rows]
"""

IMPL_TMPL = '''"""Task impl: {name}. ref is ground truth; gen is the adversarial generator."""

import numpy as np
import torch

from tasks.common import gen, ops


def ref(**inp):
{ref_body}


def gen_inputs(shape, seed):
    K, N, g, M = shape["K"], shape["N"], shape["g"], shape["M"]
{gen_body}
'''
def main():
    for name, desc, inputs, outputs, ref_body, gen_body in TASKS:
        d = os.path.join(ROOT, "tasks", "quant_pack", name)
        os.makedirs(d, exist_ok=True)
        yaml = YAML_TMPL.format(
            name=name,
            desc=textwrap.fill(desc, 96, subsequent_indent="  "),
            inputs=inputs,
            outputs=outputs,
            visible=VISIBLE,
            hidden=HIDDEN,
        )
        with open(os.path.join(d, "spec.yaml"), "w") as f:
            f.write(yaml)
        def norm(s):
            stripped = "\n".join(
                (ln[8:] if ln.startswith(" " * 8) else ln) for ln in s.splitlines()
            )
            return textwrap.indent(stripped, " " * 4)

        impl = IMPL_TMPL.format(
            name=name, ref_body=norm(ref_body), gen_body=norm(gen_body)
        )
        with open(os.path.join(d, "impl.py"), "w") as f:
            f.write(impl)
        print("wrote", name)


if __name__ == "__main__":
    main()
