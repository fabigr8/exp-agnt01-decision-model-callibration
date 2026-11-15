# One entry point per pipeline stage (protocol section 13). Requires uv.
CONFIG ?= configs/run.yaml
PY = uv run --frozen python

.PHONY: setup test data data-a data-b pilot grid analyze figures

setup:
	uv sync --frozen --all-extras

test:
	uv run --frozen pytest -q

data: data-b data-a

data-b:
	$(PY) src/prep_data.py --task B

data-a:
	$(PY) src/prep_data.py --task A

pilot grid:
	$(PY) src/run_grid.py --config $(CONFIG)

analyze figures:
	@echo "not implemented yet (plan steps 3-4); runs from stored predictions only" && exit 1
