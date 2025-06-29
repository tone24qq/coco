# tests/test_rank_cells_strict.py
import numpy as np
import pytest

import analyzer
from analyzer import rank_cells_by_prior_and_modules, compute_global_distribution

@pytest.fixture
def sample_dir(tmp_path):
    # Create sample history directory for prior calculation
    samples = tmp_path / "samples"
    samples.mkdir()
    # Single history sample with target_num at (0,1)
    data = {"rows": 2, "cols": 2, "grid": [[1, 7], [3, 4]]}
    import json, zipfile
    with zipfile.ZipFile(samples / "hist.zip", "w") as zf:
        zf.writestr("h.json", json.dumps(data))
    return str(samples)

@pytest.fixture(autouse=True)
def monkeypatch_module_scores(monkeypatch):
    # Provide two dummy modules producing contrasting scores
    def mod1(name, grid, target=None):
        arr = np.zeros_like(grid, dtype=float)
        arr[0, 0] = 1.0
        return arr

    def mod2(name, grid, target=None):
        arr = np.zeros_like(grid, dtype=float)
        arr[1, 1] = 1.0
        return arr

    # Based on module name suffix choose mod1 or mod2
    monkeypatch.setattr(analyzer, "get_module_score",
                        lambda mod, grid, target=None: mod1(mod, grid) if mod.endswith("1") else mod2(mod, grid))
    yield

def test_global_prior_influence(sample_dir):
    prior = compute_global_distribution(sample_dir, 2, 2)
    grid = np.array([[-1, 7], [3, -1]])
    mods = ["M1", "M1"]  # modules ignored when w_prior=1.0
    weights = [0.5, 0.5]
    ranks = rank_cells_by_prior_and_modules(grid, prior, mods, weights, target_num=7, w_prior=1.0)
    assert len(ranks) == 2
    # Both entries have equal probability
    assert pytest.approx(ranks[0][2], abs=1e-6) == ranks[1][2]
    total = sum(p for _, _, p in ranks)
    assert pytest.approx(100.0, rel=1e-6) == total

def test_module_scores_only(sample_dir):
    prior = compute_global_distribution(sample_dir, 2, 2)
    grid = np.array([[-1, 7], [3, -1]])
    mods = ["X1", "X2"]
    weights = [0.6, 0.4]
    ranks = rank_cells_by_prior_and_modules(grid, prior, mods, weights, target_num=7, w_prior=0.0)
    assert len(ranks) == 2
    pct_dict = {(r, c): p for r, c, p in ranks}
    assert pytest.approx(pct_dict[(0,0)], rel=1e-6) == 60.0
    assert pytest.approx(pct_dict[(1,1)], rel=1e-6) == 40.0

def test_diversity_and_sorting(sample_dir):
    prior = compute_global_distribution(sample_dir, 2, 2)
    grid = np.array([[-1, 7], [3, -1]])
    mods = ["A1", "B2"]
    weights = [0.3, 0.7]
    ranks = rank_cells_by_prior_and_modules(grid, prior, mods, weights, target_num=7, w_prior=0.2)
    ps = [p for _, _, p in ranks]
    assert pytest.approx(sum(ps), rel=1e-6) == 100.0
    assert len(set(ps)) == len(ps)
    assert ps == sorted(ps, reverse=True)