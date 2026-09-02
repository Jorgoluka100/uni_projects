"""Fail CI if protected university/course notebooks disappear or change unexpectedly."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "protected_originals.json"


def git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode("utf-8")
    return hashlib.sha1(header + data).hexdigest()


def main() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    required = payload.get("required_paths", [])
    pinned = payload.get("pinned_blob_sha", {})

    missing = [rel for rel in required if not (ROOT / rel).is_file()]
    changed = []
    for rel, expected in pinned.items():
        path = ROOT / rel
        if not path.is_file():
            continue
        actual = git_blob_sha(path)
        if actual != expected:
            changed.append({"path": rel, "expected": expected, "actual": actual})

    report = {
        "protected_originals": len(required),
        "pinned_unchanged_originals": len(pinned),
        "missing": missing,
        "unexpectedly_changed": changed,
    }
    print(json.dumps(report, indent=2))

    if missing or changed:
        raise SystemExit(
            "Protected original university/course work changed. Restore the original file, "
            "or deliberately update protected_originals.json only when the owner explicitly approves a change."
        )

    print("Protected original university/course notebooks PASSED.")


if __name__ == "__main__":
    main()
