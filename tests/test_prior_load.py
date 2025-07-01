import analyzer


def test_prior_load():
    probs = analyzer.compute_position_probabilities("", 2, 2, n_synth=20, seed=0)
    for cell in probs.values():
        assert abs(sum(cell.values()) - 1.0) < 1e-6
