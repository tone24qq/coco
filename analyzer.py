from pydantic import BaseModel
import math
import brain1, brain2, brain3, new_module

class PredictionResult(BaseModel):
    """Structured prediction result from a module."""
    row: int
    col: int
    confidence: float
    module: str
    reason: str
    predicted_value: int

# In-memory cache to store results for already-analyzed boards
_cache = {}

def analyze_board(board: list, target_number: int):
    """
    Analyze the given board to find likely positions of the target_number in hidden cells.
    Returns a list of top 3 predictions (each with coordinates, confidence, and contributing reasons).
    """
    # Create a hashable key for the board state (tuple of tuples) and target for caching
    board_key = tuple(tuple(x for x in row) for row in board)
    cache_key = (target_number, board_key)
    if cache_key in _cache:
        # Return cached result if this board-target combination was analyzed before
        return _cache[cache_key]

    predictions = []  # collect predictions from all modules

    # List of modules to run (module name, module reference)
    modules = [
        ("Brain1", brain1),
        ("Brain2", brain2),
        ("Brain3", brain3),
        ("Brain4", new_module)  # treat new_module as Brain4 for naming purposes
    ]
    for name, module in modules:
        # Each module returns a list of prediction dicts
        try:
            module_preds = module.analyze(board, target_number)
        except Exception as e:
            module_preds = []
        for pred in module_preds:
            # Ensure each prediction dict has unified keys and values
            pred['module'] = pred.get('module', name)             # module name
            pred['predicted_value'] = target_number               # target number as predicted value
            # Create a PredictionResult object for type validation and consistency
            try:
                pred_model = PredictionResult(**pred)
            except Exception as e:
                continue  # skip invalid predictions if any
            predictions.append(pred_model)

    # Combine predictions for the same cell from multiple modules
    combined_results = {}
    for pred in predictions:
        pos = (pred.row, pred.col)
        if pos not in combined_results:
            # Initialize combined result for this position
            combined_results[pos] = {
                "row": pred.row,
                "col": pred.col,
                "confidence": pred.confidence,
                "contributions": [ 
                    {"module": pred.module, "reason": pred.reason, "confidence": pred.confidence} 
                ]
            }
        else:
            # If position already has contributions, combine confidence and append reason
            existing_conf = combined_results[pos]["confidence"]
            new_conf = pred.confidence
            # Combine confidence assuming independent evidence: P = 1 - (1-p1)*(1-p2)*...
            combined_conf = 1 - (1 - existing_conf) * (1 - new_conf)
            combined_results[pos]["confidence"] = round(combined_conf, 3)
            combined_results[pos]["contributions"].append({
                "module": pred.module,
                "reason": pred.reason,
                "confidence": pred.confidence
            })

    # Remove any entries that have effectively 0 confidence (e.g., eliminated by patterns)
    combined_results = {pos: data for pos, data in combined_results.items() if data["confidence"] > 0.0}

    # Select the top 3 positions by confidence
    top3 = sorted(combined_results.values(), key=lambda x: x["confidence"], reverse=True)[:3]

    # Store result in cache and return
    _cache[cache_key] = top3
    return top3