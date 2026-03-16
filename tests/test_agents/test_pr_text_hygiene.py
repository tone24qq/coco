from __future__ import annotations

from pathlib import Path

from scripts.normalize_gitignore import normalize_gitignore


def test_normalize_gitignore_utf16_to_utf8_lf(tmp_path: Path) -> None:
    path = tmp_path / ".gitignore"
    path.write_bytes("a\r\nb\r\n\x00c\r".encode("utf-16"))

    result = normalize_gitignore(path)

    out = path.read_bytes()
    text = path.read_text(encoding="utf-8")
    assert result["has_nul_after"] is False
    assert not out.startswith(b"\xef\xbb\xbf")
    assert b"\x00" not in out
    assert text == "a\nb\nc\n"
