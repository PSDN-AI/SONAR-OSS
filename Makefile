.PHONY: help setup install test clean lint format typecheck pre-commit-install pre-commit-run check-internal-refs

help:
	@echo "SONAR-OSS: Multi-Language ASR Evaluation Toolkit"
	@echo ""
	@echo "Available commands:"
	@echo "  make setup               - Create virtual environment and install dependencies (requires uv)"
	@echo "  make install             - Install package in editable mode"
	@echo "  make test                - Run tests"
	@echo "  make clean               - Remove build artifacts and cache"
	@echo "  make lint                - Run ruff linter and format check"
	@echo "  make format              - Format code with ruff"
	@echo "  make typecheck           - Run ty type checker"
	@echo "  make pre-commit-install  - Install git pre-commit hooks (run once after clone)"
	@echo "  make pre-commit-run      - Run all pre-commit hooks against the whole repo"
	@echo "  make check-internal-refs - Fail if internal/private references are present"
	@echo ""
	@echo "Quick start:"
	@echo "  make setup"
	@echo "  source .venv/bin/activate"

setup:
	@echo "Installing frozen environment with uv..."
	uv sync --frozen --extra dev
	@echo ""
	@echo "Setup complete! Activate with:"
	@echo "  source .venv/bin/activate"

install:
	uv pip install -e ".[dev]"

test:
	uv run pytest tests/ -q

clean:
	rm -rf build/ dist/ *.egg-info
	rm -rf .venv/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true

lint:
	uv run ruff check psdn_sonar tests
	uv run ruff format --check psdn_sonar tests

format:
	uv run ruff format psdn_sonar tests
	uv run ruff check psdn_sonar tests --fix

typecheck:
	uv run ty check psdn_sonar/

pre-commit-install:
	pre-commit install

pre-commit-run:
	pre-commit run --all-files

check-internal-refs:
	./scripts/check_internal_refs.sh
