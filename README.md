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
provided grid. Parameters such as rank and learning rate can be tuned via
keyword arguments to `predict`.

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
