from __future__ import annotations

from pathlib import Path

DECODE_ORDER = ("utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be")


def decode_with_fallback(raw: bytes) -> str:
    for enc in DECODE_ORDER:
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def normalize_text(text: str) -> str:
    normalized = text.replace("\x00", "")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    return normalized


def normalize_gitignore(path: Path) -> dict:
    raw = path.read_bytes()
    decoded = decode_with_fallback(raw)
    normalized = normalize_text(decoded)
    path.write_text(normalized, encoding="utf-8", newline="\n")

    post = path.read_bytes()
    return {
        "path": str(path),
        "had_nul_before": b"\x00" in raw,
        "has_nul_after": b"\x00" in post,
        "utf8_bom_after": post.startswith(b"\xef\xbb\xbf"),
    }


def main() -> None:
    result = normalize_gitignore(Path(".gitignore"))
    if result["has_nul_after"]:
        raise SystemExit(".gitignore still contains NUL bytes after normalization")


if __name__ == "__main__":
    main()
