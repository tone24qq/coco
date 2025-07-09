import brain


def test_get_core_modules_limit(monkeypatch):
    monkeypatch.setenv("CORE_LIMIT", "5")
    mods = brain.get_core_modules()
    assert len(mods) == 5
    monkeypatch.delenv("CORE_LIMIT", raising=False)


def test_get_core_modules_order():
    mods = brain.get_core_modules(limit=6)
    sorted_mods = sorted(brain.AGG_WEIGHTS, key=brain.AGG_WEIGHTS.get, reverse=True)[:6]
    assert mods == sorted_mods


def test_get_core_modules_warn_invalid_env(monkeypatch, caplog):
    monkeypatch.setenv("CORE_LIMIT", "bad")
    with caplog.at_level("WARNING"):
        brain.get_core_modules()
    assert any("Invalid CORE_LIMIT" in r.message for r in caplog.records)
    monkeypatch.delenv("CORE_LIMIT", raising=False)


def test_get_core_modules_invalid_env_default(monkeypatch):
    monkeypatch.setenv("CORE_LIMIT", "bad")
    mods = brain.get_core_modules()
    assert len(mods) == 6
    monkeypatch.delenv("CORE_LIMIT", raising=False)


def test_get_core_modules_negative_limit():
    mods = brain.get_core_modules(limit=-2)
    assert len(mods) == 1


def test_get_core_modules_env_negative(monkeypatch):
    monkeypatch.setenv("CORE_LIMIT", "-3")
    mods = brain.get_core_modules()
    assert len(mods) == 1
    monkeypatch.delenv("CORE_LIMIT", raising=False)


def test_get_core_modules_invalid_param(monkeypatch, caplog):
    monkeypatch.setenv("CORE_LIMIT", "4")
    with caplog.at_level("WARNING"):
        mods = brain.get_core_modules(limit="bad")  # type: ignore[arg-type]
    assert len(mods) == 4
    assert any("Invalid limit" in r.message for r in caplog.records)
    monkeypatch.delenv("CORE_LIMIT", raising=False)
