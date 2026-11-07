CONDA ?= /Users/zenith/miniconda3/bin/conda
ENV   ?= qkb
RUN   := $(CONDA) run --no-capture-output -n $(ENV)

.PHONY: setup test tasks-validate lint

setup:
	$(CONDA) env create -f environment.yml || $(CONDA) env update -n $(ENV) -f environment.yml --prune

test: tasks-validate
	$(RUN) python -m pytest -q

tasks-validate:
	$(RUN) python harness/validate_tasks.py

lint:
	$(RUN) python -m compileall -q $$(git ls-files '*.py')
