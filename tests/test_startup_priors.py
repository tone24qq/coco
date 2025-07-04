import asyncio
import json
from pathlib import Path

import app
import brain


def test_find_closest_prior_key():
    brain.priors_map = {"2x2": {"a": 1}, "4x4": {"b": 1}}
    key = app.find_closest_prior_key((3, 3), brain.priors_map)
    assert key == "2x2"
    prior = app.get_prior_for_shape(3, 3)
    assert prior == brain.priors_map["2x2"]


def test_load_priors_async_file(tmp_path, monkeypatch):
    p = tmp_path / "priors_2x2.json"
    p.write_text(json.dumps({"1": 1.0}))
    monkeypatch.setattr(app, "PRIORS_DIR", tmp_path)
    priors = asyncio.run(app.load_priors_async())
    assert "2x2" in priors


def test_load_priors_async_fallback(monkeypatch):
    monkeypatch.setattr(app, "PRIORS_DIR", Path("nope"))
    monkeypatch.setattr(app, "_load_priors_files", lambda: {"5x5": {"x": 1}})
    priors = asyncio.run(app.load_priors_async())
    assert priors == {"5x5": {"x": 1}}
