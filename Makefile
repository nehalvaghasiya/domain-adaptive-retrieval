.PHONY: help install fmt format format-check lint type test check pre-commit smoke build notebooks lock

SOURCES := src tests examples scripts

help:
	@echo 'install fmt format format-check lint type test check smoke notebooks build lock'
install:
	uv sync --group dev

fmt format:
	uv run ruff check --fix .
	uv run ruff format .

format-check:
	uv run ruff format --check $(SOURCES)

lint:
	uv run ruff check $(SOURCES)

type:
	uv run mypy src/domain_adaptive_retrieval

test:
	uv run pytest -m 'not gpu and not network and not slow' -q --cov=domain_adaptive_retrieval --cov-report=term-missing --cov-fail-under=85

check: format-check lint type test notebooks

pre-commit:
	uv run pre-commit run --all-files

smoke:
	uv run dar smoke --offline --config configs/experiments/smoke.yaml

notebooks:
	uv run python scripts/validate_notebooks.py

build:
	uv build
	uv run twine check dist/*

lock:
	uv lock
