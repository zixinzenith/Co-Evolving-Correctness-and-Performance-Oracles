# agent/

The generation loop: a fixed 6-iteration loop per task (generate → compile/run → verifier feedback → revise), identical across models so results are comparable. Prompt variants live in `prompts/`.
