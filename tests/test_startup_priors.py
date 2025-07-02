import asyncio
import json
from pathlib import Path

import app
import brain


def test_find_closest_prior_key():
    brain.priors_map = {(2, 2): {"a": 1}, (4, 4): {"b": 1}}
    key = app.find_closest_prior_key((3, 3), brain.priors_map)
    assert key == (2, 2)
    prior = app.get_prior_for_shape(3, 3)
    assert prior == brain.priors_map[(2, 2)]


def test_load_priors_async_file(tmp_path, monkeypatch):
    path = tmp_path / "p.json"
    path.write_text(json.dumps({"2x2": {}}))
    monkeypatch.setattr(app, "PRIORS_PATH", path)
    priors = asyncio.run(app.load_priors_async())
    assert (2, 2) in priors


def test_load_priors_async_fallback(monkeypatch):
    monkeypatch.setattr(app, "PRIORS_PATH", Path("nope.json"))
    monkeypatch.setattr(app, "_build_default_prior", lambda: {(5, 5): {"x": 1}})
    priors = asyncio.run(app.load_priors_async())
    assert priors == {(5, 5): {"x": 1}}
