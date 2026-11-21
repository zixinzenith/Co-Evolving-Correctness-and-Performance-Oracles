CONDA ?= /Users/zenith/miniconda3/bin/conda
ENV   ?= qkb
RUN   := $(CONDA) run --no-capture-output -n $(ENV)

.PHONY: setup test tasks-validate mutants run-e2e lint

# Create the pinned CPU-only env from environment.yml (update it if it already exists).
setup:
	$(CONDA) env create -f environment.yml || $(CONDA) env update -n $(ENV) -f environment.yml --prune

test: tasks-validate
	$(RUN) python -m pytest -q

tasks-validate:
	$(RUN) python harness/validate_tasks.py

# Oracle adequacy: run the mutant suite against the verifier (must be 570/570).
mutants:
	$(RUN) python mutants/run_mutants.py

# Single-task end-to-end through the generation loop with the scripted fake model.
run-e2e:
	$(RUN) python harness/run.py --task quant_pack/int4_quantize_asym --condition unittest_feedback --model fake_buggy

# Minimal lint: byte-compile every tracked Python file (syntax check, no extra deps).
lint:
	$(RUN) python -m compileall -q $$(git ls-files '*.py')
