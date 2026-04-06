# Position Backtest Data Format

Required fields per sample:

- `sample_id`: string
- `grid`: 2D integer array (`-1` for missing cell)
- `answer_row`: integer answer row index (0-based or 1-based; loader auto-detects)
- `answer_col`: integer answer col index (0-based or 1-based; loader auto-detects)
- `source`: source name

Optional fields:

- `order_index`: integer sequence index for walk-forward split
- `answer_value`: integer value at answer position

Validation rules (fail-fast):

1. `grid` must be 2D.
2. answer coordinate must be in-bound after coordinate mode normalization.
3. answer position in `grid` must be `-1`.
4. when all samples are invalid, run exits with error.
