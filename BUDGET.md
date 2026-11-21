# Budget

Phase 0 spend so far: **$0** — CPU-only development on the local machine.

| Item | Estimate | Notes |
|---|---|---|
| LLM API spend | $250–400 | 40 tasks × 5 models × 4 conditions × 6 iterations ≈ 4,800 attempts at 10–15k tokens each (prompt incl. format card + generated code + feedback); re-priced at Phase 1 start |
| GPU rental (Phase 1, Ada SM89: 4090 / L40S) | $40–70 | ~$0.5–0.9/h × ~80 h: expert baselines, hidden-shape eval, performance sweep |
| GPU rental (optional Hopper/Blackwell phase) | $150–250 | H100 for NVFP4/MXFP4 tasks and the DeepGEMM baseline, only if pursued |
| Disk | ~4 GB | CPU PyTorch env (`qkb`, ~2 GB) + `runs/` JSONL records and logs |

Core Phase 1 total: **≈ $300–470**.
