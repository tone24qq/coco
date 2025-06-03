import json
from analyzer import analyze_board

if __name__ == "__main__":
    # Example usage: define a sample board and target number.
    # Hidden squares are represented by None.
    sample_board = [
        [12,  5, None,  8, 19],
        [None, 15, 14, 13, None],
        [ 7, None, 16, None, 18],
        [None, 11, 10,  9, None],
        [ 2, None,  3, None,  4]
    ]
    target_number = 17

    # Analyze the sample board for the target number's likely positions
    top3_predictions = analyze_board(sample_board, target_number)

    # Output the Top 3 predictions in readable JSON format
    print("Top 3 Predictions:")
    print(json.dumps(top3_predictions, ensure_ascii=False, indent=2))