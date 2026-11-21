# Status

## Phase 0 — task set and verifier on CPU (complete)

- [x] 0.1 Novelty check — gap confirmed (`notes/novelty_check.md`)
- [x] 0.2 Related work review (`notes/related_work.md`)
- [x] 0.3 Repo skeleton and pinned CPU environment — `qkb` env: Python 3.11.18, torch 2.15.0 (CPU), numpy 2.4.6, pytest 9.2.0
- [x] 0.4 Task specs — 40 operators across 5 groups: 12 quant_pack, 8 dequant_gemv, 6 gemm, 7 kv_cache, 7 fused_longtail; all validated (`tasks/manifest.json`, `make test`)
- [x] 0.5 Format docs — INT4 packing (AWQ vs GPTQ, sha256-pinned sources), FP8 e4m3/e5m2 (empirically verified in torch 2.15.0), rounding & zero-point conventions
- [x] 0.6 Verifier — bit-exact + format-derived tolerances + anti-cheat (forbidden imports, output aliasing, hidden shapes); oracle unit-tested (`verifier/`)
- [x] 0.7 Mutation testing of the verifier — 28 mutants over the documented bug axes, 570/570 detected (`make mutants`)
- [x] 0.8 Agent loop and prompts — fixed 6-iteration loop, 4 prompt conditions, fake-model e2e tests (`agent/`, `prompts/`)
- [x] 0.9 Budget (`BUDGET.md`)

## Phase 1 — performance and study runs on rented Ada GPU (pending)

- [ ] Rent RTX 4090 / L40S (SM89) instance; re-verify fp8 cast / `torch._scaled_mm` behavior there (flag divergences from `formats/fp8_e4m3_e5m2.md`)
- [ ] Move candidate execution into per-attempt subprocesses; wire real model clients behind the `run.py` model interface
- [ ] Expert baselines on GPU: Marlin, vLLM Humming, torchao, bitsandbytes, `torch._scaled_mm`
- [ ] RQ1 runs: 5 models × 40 tasks × 4 conditions × 6 iterations
- [ ] RQ2: error taxonomy from recorded failures (`runs/` JSONL)
- [ ] RQ3: intervention analysis across prompt conditions

Optional Hopper/Blackwell stage: NVFP4/MXFP4 tasks, DeepGEMM baseline (SM90+).
