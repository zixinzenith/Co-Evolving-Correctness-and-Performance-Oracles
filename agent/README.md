# agent/

The generation loop (`loop.py`): a fixed 6-iteration cycle per task — generate → run against
visible shapes → verifier feedback → revise — identical across models and conditions so
iteration counts stay comparable. Candidates that import the oracle stack or return their
inputs aliased are rejected (`verifier/anticheat.py`).

`fake_model.py` provides scripted models (good / buggy-then-good / always-buggy) for the
end-to-end tests in `tests/` — the full pipeline runs without any API spend. Phase 1 plugs
real API clients into the same callable interface.
