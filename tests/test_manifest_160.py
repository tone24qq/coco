from pathlib import Path

from src.board_manifest import build_multisize_manifest


def test_manifest_supports_160(tmp_path: Path) -> None:
    folder = tmp_path / "gogo" / "160"
    folder.mkdir(parents=True)
    (folder / "SAMPLE160.jpg").write_bytes(b"x")
    entries, audit = build_multisize_manifest(tmp_path)
    size_160 = [e for e in entries if e.size_class == "160"]
    assert len(size_160) == 1
    assert size_160[0].valid is True
    assert audit.total_images == 1
