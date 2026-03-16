from __future__ import annotations

import subprocess
from fnmatch import fnmatch

GENERATED_GUARD_PATTERNS = [
    "data/feature_store/*",
    "data/processed/bingo_draws_canonical.csv",
    "data/processed/bingo_draws_canonical.parquet",
    "data/processed/history_snapshot.parquet",
    "data/processed/history_snapshot_meta.json",
    "data/raw/raw_manifest.json",
]


def _run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)


def _changed_paths() -> list[str]:
    out = _run(["git", "diff", "--name-only"]).splitlines()
    staged = _run(["git", "diff", "--cached", "--name-only"]).splitlines()
    return sorted(set([p.strip() for p in out + staged if p.strip()]))


def _has_binary_patch_for(path: str) -> bool:
    diff = _run(["git", "diff", "--", path])
    staged = _run(["git", "diff", "--cached", "--", path])
    merged = f"{diff}\n{staged}"
    return "GIT binary patch" in merged or "Binary files" in merged


def run_checks() -> None:
    changed = _changed_paths()
    failed: list[str] = []

    for path in changed:
        if _has_binary_patch_for(path):
            failed.append(f"binary-patch-detected: {path}")

    for path in changed:
        if any(fnmatch(path, pattern) for pattern in GENERATED_GUARD_PATTERNS):
            failed.append(f"generated-artifact-in-diff: {path}")

    if failed:
        details = "\n".join(failed)
        raise SystemExit(f"pre_pr_checks failed:\n{details}")


if __name__ == "__main__":
    run_checks()
