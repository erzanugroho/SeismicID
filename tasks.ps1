# PowerShell helper script for Windows users without `make` installed.
# Usage: .\tasks.ps1 <target>
# Targets: install, install-dev, lint, format, test, test-cov, run, clean

param(
    [Parameter(Mandatory=$true, Position=0)]
    [ValidateSet("install","install-dev","lint","format","test","test-cov","run","clean")]
    [string]$Target
)

$ErrorActionPreference = "Stop"

switch ($Target) {
    "install"     { python -m pip install -r requirements.txt }
    "install-dev" {
        python -m pip install -r requirements-dev.txt
        try { pre-commit install } catch { Write-Warning "pre-commit not initialized (no git repo?)" }
    }
    "lint" {
        ruff check backend
        mypy backend/app
    }
    "format" {
        ruff format backend
        ruff check --fix backend
    }
    "test"     { pytest backend/tests }
    "test-cov" { pytest backend/tests --cov=backend/app --cov-report=term-missing }
    "run"      { uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload }
    "clean" {
        Remove-Item -Recurse -Force .pytest_cache, .ruff_cache, .mypy_cache, htmlcov, .coverage -ErrorAction SilentlyContinue
        Get-ChildItem -Path . -Filter __pycache__ -Recurse -Directory | Remove-Item -Recurse -Force
    }
}
