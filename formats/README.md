# formats/

Quantization format specifications (AWQ/GPTQ packing, FP8 e4m3/e5m2 rules, scale/zero-point layouts, rounding modes, ...).

Each spec records the upstream source it was derived from, with repository URL and commit hash, so the definition is reproducible. These specs are also the source for the "format spec card" prompt condition in `prompts/`.
