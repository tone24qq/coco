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
