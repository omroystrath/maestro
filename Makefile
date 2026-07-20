# MAESTRO developer tasks. Run `make help` for a list.
.DEFAULT_GOAL := help
PY ?= python3
VENV := .venv
BIN := $(VENV)/bin

.PHONY: help setup install lint fmt type test smoke clean

help: ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
	awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## create venv and install the package (editable) with dev extras
	$(PY) -m venv $(VENV)
	$(BIN)/pip install -U pip
	$(BIN)/pip install -e ".[dev,sci]"
	@echo "Activate with: source $(VENV)/bin/activate"

install-train: ## add the training (torch) extra
	$(BIN)/pip install -e ".[train]"

lint: ## ruff + black --check
	$(BIN)/ruff check src tests
	$(BIN)/black --check src tests

fmt: ## auto-format
	$(BIN)/ruff check --fix src tests
	$(BIN)/black src tests

type: ## mypy
	$(BIN)/mypy

test: ## run the test suite
	$(BIN)/pytest

smoke: ## end-to-end pipeline on tiny analytic data (needs the train extra)
	$(BIN)/maestro info
	$(BIN)/maestro simulate --config configs/data/toy.yaml --out data/interim/toy
	$(BIN)/maestro gen train  --config configs/model/generator_small.yaml
	$(BIN)/maestro fm  train  --config configs/model/foundation_small.yaml
	$(BIN)/maestro eval        --config configs/train/eval_smoke.yaml

clean: ## remove caches and build artefacts
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
