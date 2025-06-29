# Scratch-Card Prediction System

This project provides a FastAPI service and CLI tool for predicting hidden numbers on scratch cards. The system uses modular heuristics and Monte-Carlo simulation to estimate probabilities.

## Installation

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # optional, for development
```

## Running Tests

Execute all unit tests with:

```bash
pytest -q
```

## Continuous Integration

GitHub Actions run linting and the test suite on every push and pull request. Slow tests are executed separately on a scheduled or manual trigger.

