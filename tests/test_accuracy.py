import analyzer


def test_evaluate_prediction_accuracy():
    acc = analyzer.evaluate_prediction_accuracy(num_trials=3, seed=0)
    assert 0.0 <= acc <= 1.0
