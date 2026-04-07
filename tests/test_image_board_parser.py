from src.board_manifest import ManifestEntry
from src.image_board_parser import parse_manifest_entries


def test_parser_handles_invalid_manifest_entry() -> None:
    manifest = [
        ManifestEntry(
            sample_id="bad",
            size_class="120",
            image_paths=["not_found.jpg"],
            page_count=1,
            source_folder=".",
            valid=False,
            invalid_reason="120_missing_page",
        )
    ]
    boards, audits, pending, _ = parse_manifest_entries(manifest, min_confidence=0.9)
    assert boards == []
    assert audits[0].invalid_reason == "120_missing_page"
    assert pending[0]["reason"] == "120_missing_page"
