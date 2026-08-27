"""Validate retained evidence for the production-style projects under projects/."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = {
    "CareerLens AI": ROOT / "verified" / "careerlens_ai" / "verification.json",
    "ExperimentLab": ROOT / "verified" / "experiment_lab" / "verification.json",
    "ModelWatch": ROOT / "verified" / "model_watch" / "verification.json",
    "GroundedRAG": ROOT / "verified" / "grounded_rag" / "verification.json",
    "Reliable Event Pipeline": ROOT / "projects" / "reliable_event_pipeline" / "results" / "verified_run.json",
}


def main() -> int:
    failures: list[str] = []
    for name, path in EVIDENCE.items():
        if not path.is_file():
            failures.append(f"{name}: missing {path.relative_to(ROOT)}")
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            failures.append(f"{name}: invalid JSON ({exc})")
            continue
        if payload.get("verification_pass") is not True:
            failures.append(f"{name}: verification_pass is not true")
            continue
        print(f"PASS: {name} -> {path.relative_to(ROOT)}")

    if failures:
        for item in failures:
            print(f"FAIL: {item}")
        return 1

    print(f"New project evidence gate passed for {len(EVIDENCE)} projects.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
