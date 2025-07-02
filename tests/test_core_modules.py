import brain


def test_get_core_modules_limit(monkeypatch):
    monkeypatch.setenv("CORE_LIMIT", "5")
    mods = brain.get_core_modules()
    assert isinstance(mods, list)
    assert len(mods) == 5
    monkeypatch.delenv("CORE_LIMIT", raising=False)


def test_get_core_modules_order():
    mods = brain.get_core_modules(limit=6)
    expected = sorted(brain.AGG_WEIGHTS, key=brain.AGG_WEIGHTS.get, reverse=True)[:6]
    assert mods == expected


def test_get_core_modules_warn_invalid_env(monkeypatch, caplog):
    monkeypatch.setenv("CORE_LIMIT", "bad")
    with caplog.at_level("WARNING"):
        mods = brain.get_core_modules()
    assert isinstance(mods, list)
    assert len(mods) == 6  # fallback default
    assert any("Invalid CORE_LIMIT" in r.message for r in caplog.records)
    monkeypatch.delenv("CORE_LIMIT", raising=False)


def test_get_core_modules_invalid_env_default(monkeypatch):
    monkeypatch.setenv("CORE_LIMIT", "bad")
    mods = brain.get_core_modules()
    assert isinstance(mods, list)
    assert len(mods) == 6  # fallback to default
    monkeypatch.delenv("CORE_LIMIT", raising=False)