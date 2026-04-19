from __future__ import annotations

from types import SimpleNamespace

import pytest

import scripts.run_training_pipeline as p


def test_decode_exit_code_stack_overflow() -> None:
    msg = p._decode_exit_message(3221225725)
    assert "0xC00000FD" in msg


def test_python_guard_blocks_314(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(p.sys, "version_info", SimpleNamespace(major=3, minor=14))
    with pytest.raises(RuntimeError):
        p._check_python_supported(False)
    p._check_python_supported(True)
