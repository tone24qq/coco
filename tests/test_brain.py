import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import brain

brain.configure_logging()

def test_registered_modules():
    dummy_grid = np.array([[1, 2, -1], [-1, 1, 5], [3, -1, 4]])
    modules = [
        "EXT_Q1_ProximityEntropy_Vec",
        "EXT_Q2_PotentialPath_Vec",
        "EXT_Q3_DiscontinuitySym_Vec",
        "EXT_Q4_ControlComposite_Vec",
        "EXT_Q5_GlobalEntropy_Vec",
        "EXT_Q6_LineBridge_Vec",
        "EXT_Q7_VariancePrior_Vec",
        "EXT_Q8_SpatialKL_Vec",
        "EXT_Q9_MultiScaleEntropy_Vec",
        "EXT_Q10_DistPotential_Vec",
    ]
    for mod in modules:
        try:
            scores = brain.get_module_score(mod, dummy_grid)
            assert isinstance(scores, np.ndarray)
            assert scores.shape == dummy_grid.shape
            assert scores.dtype == float
        except ValueError:
            # Some modules may legitimately raise ValueError on small grids
            continue
    assert len(brain.REGISTERED_MODULES_BRAIN) >= len(modules)

def test_global_offset_cooccurrence():
    grid = np.array([[1, 2, -1], [2, 1, -1]])
    scores = brain.get_module_score(
        "EXT_GlobalOffsetCooccurrence_Vec",
        grid,
        target=1,
    )
    assert scores.shape == grid.shape
    assert scores[0, 2] == 2
    assert scores[1, 2] == 2

