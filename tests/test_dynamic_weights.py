import importlib

import brain


def test_dynamic_weight_loading(tmp_path, monkeypatch):
    perf = tmp_path / "perf.txt"
    perf.write_text("EXT_Q1_ProximityEntropy_Vec 0.8\nEXT_Q2_PotentialPath_Vec 0.2\n")
    monkeypatch.setenv("PERFORMANCE_FILE", str(perf))
    monkeypatch.setenv("MIN_WEIGHT", "0.05")
    mod = importlib.reload(brain)
    assert abs(sum(mod.AGG_WEIGHTS.values()) - 1.0) < 1e-6
    assert (
        mod.AGG_WEIGHTS["EXT_Q1_ProximityEntropy_Vec"]
        > mod.AGG_WEIGHTS["EXT_Q2_PotentialPath_Vec"]
    )
    monkeypatch.delenv("PERFORMANCE_FILE", raising=False)
    monkeypatch.delenv("MIN_WEIGHT", raising=False)
    importlib.reload(brain)
