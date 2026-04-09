from __future__ import annotations

from fastapi import FastAPI, HTTPException

from src.inference_models import InferTargetPositionRequest, InferTargetPositionResponse
from src.inference_service import (
    build_cell_candidates,
    build_explanation,
    compute_remaining_numbers,
    parse_board_input,
    rank_candidates,
    score_candidates,
    validate_target_number,
)

app = FastAPI(title="Scratchcard Board Inference Service", version="v1")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/infer_target_position", response_model=InferTargetPositionResponse)
def infer_target_position(payload: InferTargetPositionRequest) -> InferTargetPositionResponse:
    try:
        parsed = parse_board_input(payload.board)
        remaining = compute_remaining_numbers(parsed)
        status, opened_cell = validate_target_number(payload.target_number, parsed, remaining)

        unopened_cells_payload = [
            {"row": r + 1, "col": c + 1} for r, c in parsed.unopened_cells
        ]

        if status == "already_opened" and opened_cell is not None:
            reasoning = [
                f"盤面總格數為 {parsed.rows * parsed.cols}，合法數字集合為 1..{parsed.rows * parsed.cols}",
                f"target_number={payload.target_number} 已經在已開格",
            ]
            return InferTargetPositionResponse(
                status="already_opened",
                board_shape={"rows": parsed.rows, "cols": parsed.cols},
                target_number=payload.target_number,
                remaining_numbers=remaining,
                unopened_cells=unopened_cells_payload,
                best_cell={"row": opened_cell[0] + 1, "col": opened_cell[1] + 1, "score": 1.0},
                candidate_cells=[],
                confidence_score=1.0,
                reasoning=reasoning,
                module_contributions={},
                metadata={
                    "score_type": "position_confidence",
                    "source": payload.source,
                    "version": "v1",
                },
            )

        if not parsed.unopened_cells:
            raise ValueError("board has no unopened cells")

        candidates = build_cell_candidates(parsed.unopened_cells)
        scored, weights, module_explanations = score_candidates(
            payload.board,
            candidates,
            payload.target_number,
        )
        ranked = rank_candidates(scored)

        best = ranked[0]
        reasoning = build_explanation(
            parsed.rows,
            parsed.cols,
            payload.target_number,
            remaining,
            len(parsed.unopened_cells),
            weights,
            best["cell"],
            module_explanations,
        )

        candidate_cells = [
            {
                "row": cell["cell"][0] + 1,
                "col": cell["cell"][1] + 1,
                "score": round(float(cell["score"]), 6),
                "module_scores": {
                    k: round(float(v), 6)
                    for k, v in sorted(cell["module_scores"].items())
                },
            }
            for cell in ranked
        ]

        return InferTargetPositionResponse(
            status="ok",
            board_shape={"rows": parsed.rows, "cols": parsed.cols},
            target_number=payload.target_number,
            remaining_numbers=remaining,
            unopened_cells=unopened_cells_payload,
            best_cell={
                "row": best["cell"][0] + 1,
                "col": best["cell"][1] + 1,
                "score": round(float(best["score"]), 6),
            },
            candidate_cells=candidate_cells,
            confidence_score=round(float(best["score"]), 6),
            reasoning=reasoning,
            module_contributions=weights,
            metadata={
                "score_type": "position_confidence",
                "source": payload.source,
                "version": "v1",
            },
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
