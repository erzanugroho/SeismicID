.PHONY: install install-dev lint format test test-cov run clean

PYTHON ?= python
UVICORN ?= uvicorn

install:
	$(PYTHON) -m pip install -r requirements.txt

install-dev:
	$(PYTHON) -m pip install -r requirements-dev.txt
	pre-commit install

lint:
	ruff check backend
	mypy backend/app

format:
	ruff format backend
	ruff check --fix backend

test:
	pytest backend/tests

test-cov:
	pytest backend/tests --cov=backend/app --cov-report=term-missing

run:
	$(UVICORN) backend.app.main:app --host 0.0.0.0 --port 8000 --reload

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
