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

## Indexing Convention

All API responses and configuration parameters use **1-based** row/column
indices. Internally, algorithms still operate with NumPy's 0-based indexing. For
example, a prediction for the top-left cell will be returned as `(row=1, col=1)`.

## Continuous Integration

GitHub Actions run linting and the test suite on every push and pull request. Slow tests are executed separately on a scheduled or manual trigger.

## Deployment on Render

To reduce build time when manually deploying to Render:

1. Switch your Render service to **Docker deploy** and use the provided `Dockerfile`.
   Because dependencies are installed in an earlier layer, they will be reused as long as `requirements.txt` remains unchanged.
2. Enable **Build Cache** under **Settings → Build & Deploy** and add the following path:

```
~/.cache/pip
```

Render will restore this cache before each build and save it afterwards, avoiding repeated downloads.

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

## Indexing Convention

All API responses and configuration parameters use **1-based** row/column
indices. Internally, algorithms still operate with NumPy's 0-based indexing. For
example, a prediction for the top-left cell will be returned as `(row=1, col=1)`.

## Continuous Integration

GitHub Actions run linting and the test suite on every push and pull request. Slow tests are executed separately on a scheduled or manual trigger.

## Deployment on Render

To reduce build time when manually deploying to Render:

1. Switch your Render service to **Docker deploy** and use the provided `Dockerfile`.
   Because dependencies are installed in an earlier layer, they will be reused as long as `requirements.txt` remains unchanged.
2. Enable **Build Cache** under **Settings → Build & Deploy** and add the following path:

```
~/.cache/pip
```

Render will restore this cache before each build and save it afterwards, avoiding repeated downloads.

## Reliability & Accuracy Evaluation

This repository includes a suite of stress tests to verify the **actual accuracy and reliability** of prediction models.

### Key Components

- `tests/reliability_utils.py`  
  Core testing logic. Supports:
  - `run_infinite_test()` — Infinite simulation to observe accuracy drift.
  - `run_until_converged()` — Batches simulation until confidence interval < δ.

- `tests/test_reliable_accuracy.py`  
  Runs 1000 randomized trials for several board sizes. Useful for baseline comparison.

- `tests/test_accuracy_converges.py`  
  Verifies that the accuracy converges to a statistically valid estimate.

- `tests/reliable_accuracy_test_suite.py`  
  Lightweight wrapper for one-off accuracy testing.

### Example Usage

Run 1000 randomized trials to estimate accuracy:

```bash
python -m tests.reliable_accuracy_test_suite
```

Run stress test with live accuracy output:

```bash
python -m tests.reliability_utils run_infinite_test
```

Run convergence validation (used in CI):

```bash
pytest tests/test_accuracy_converges.py
```


