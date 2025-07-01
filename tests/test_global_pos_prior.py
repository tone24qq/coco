import analyzer
from position_prior import build_position_prior


def test_global_position_prior(tmp_path):
    out = tmp_path / "pos_freq.npz"
    build_position_prior(2, 2, str(out), buckets=2, n_synth=20, seed=0)

    analyzer.load_global_pos_freq(str(tmp_path))
    probs = analyzer.compute_position_probabilities(str(tmp_path), 2, 2)
    for cell in probs.values():
        assert abs(sum(cell.values()) - 1.0) < 1e-6
