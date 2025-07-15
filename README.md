# Matrix Factorization Predict Agent

This repository demonstrates a simple approach for reconstructing masked scratch
card grids. The `predict_agent` module applies low rank matrix factorization to
estimate the hidden values, then returns candidate cell locations for a target
number.

## Strategy

1. **Matrix Completion** – Missing values marked as `-1` are first replaced by
the mean of their respective row. The algorithm then performs gradient descent
on a low rank factorization `U @ V` using only observed entries.
2. **Candidate Ranking** – After reconstruction, the absolute difference between
the predicted value and the requested `target` number is computed for every
cell. Cells are returned sorted by this difference with scores
`1 / (1 + diff)`.

This method does not rely on memorized boards and only uses the structure of the
provided grid. Parameters such as rank, learning rate and the optional
`max_val` bound can be tuned via keyword arguments to `predict`.

## Testing

`pytest` includes a regression test that reconstructs the challenge board and
confirms the output shape and value range. Because the algorithm is heuristic,
the test does not enforce a specific accuracy but ensures execution stability.

## Service

A FastAPI service exposes the predictor via `/predict` and a health-check root
endpoint. Run locally using Uvicorn or the FastAPI CLI:

```bash
uvicorn coco_service.main:app --reload
# or
fastapi dev coco_service/main.py
```

Send a request:

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"board": [[1, -1], [3, 4]], "target": 2}'
```

### Docker

Build and run using Docker:

```bash
docker build -t predictor .
docker run -p 8000:8000 predictor
```

## Installation

Install the package locally using pip:

```bash
pip install .
```

## RandomForest Inference Utility

The `rf_infer` package loads a trained RandomForest model and predicts the best
cells for a target value. Install the package and run `rf-infer`:

```bash
rf-infer --input boards/2x2_input.json --output results/out.json --k 3
```

`boards/4x5_input.json` should contain either a single object or a list of
objects with the fields:

```json
{
  "board": [[1, -1], [3, -1]],
  "target": 4
}
```

The output JSON will be a list of results with the structure:

```json
[
  {
    "rows": 2,
    "cols": 2,
    "target": 4,
    "predictions": [{"r": 1, "c": 1, "prob": 0.75}]
  }
]
```

A helper Dockerfile (`Dockerfile.rf`) builds a minimal image with the CLI.

Sample models are provided under `models/` along with example boards in
`boards/`. Try the following command to see the CLI in action:

```bash
rf-infer --input boards/2x2_input.json --output results/out.json --k 2
```
