"""Evidence/safety gate for restored healthcare AI projects.

This does not claim clinical validity. It validates a small JSON evidence contract so
health-related portfolio work cannot be promoted without patient/group separation,
source provenance, intended/non-intended use, held-out evaluation, uncertainty/error
analysis and human-oversight limitations.

Use for the Multi-Modal Health Analytics and X-ray projects after a clean rerun.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

REQUIRED_KEYS={"project","source","source_date_or_version","group_key","train_groups","validation_groups","test_groups","intended_use","non_intended_use","heldout_metrics","uncertainty_or_abstention","limitations","human_oversight"}


def validate(payload: dict[str,Any]) -> list[str]:
    errors=[]; missing=sorted(REQUIRED_KEYS-set(payload))
    if missing: errors.append("missing fields: "+", ".join(missing)); return errors
    group_sets={name:set(map(str,payload.get(name,[]))) for name in ("train_groups","validation_groups","test_groups")}
    if not all(group_sets.values()): errors.append("train/validation/test group lists must all be non-empty")
    if group_sets["train_groups"] & group_sets["test_groups"]: errors.append("group leakage: train intersects test")
    if group_sets["validation_groups"] & group_sets["test_groups"]: errors.append("group leakage: validation intersects test")
    if group_sets["train_groups"] & group_sets["validation_groups"]: errors.append("group leakage: train intersects validation")
    if not str(payload.get("source","")).strip(): errors.append("source provenance is empty")
    if not str(payload.get("source_date_or_version","")).strip(): errors.append("source date/version is empty")
    if len(payload.get("non_intended_use",[]))<2: errors.append("document at least two non-intended uses")
    if len(payload.get("limitations",[]))<3: errors.append("document at least three limitations")
    metrics=payload.get("heldout_metrics",{})
    if not isinstance(metrics,dict) or not metrics: errors.append("held-out metrics are missing")
    uncertainty=payload.get("uncertainty_or_abstention",{})
    if not isinstance(uncertainty,dict) or not uncertainty: errors.append("uncertainty/abstention evidence is missing")
    if not str(payload.get("human_oversight","")).strip(): errors.append("human oversight statement is empty")
    intended=str(payload.get("intended_use","")).lower()
    if any(term in intended for term in ("autonomous diagnosis","replace clinician","clinical deployment")): errors.append("intended use overclaims clinical deployment")
    return errors


def self_test() -> None:
    good={"project":"demo","source":"public research dataset","source_date_or_version":"v1","group_key":"patient_id","train_groups":["p1","p2"],"validation_groups":["p3"],"test_groups":["p4"],"intended_use":"educational retrospective research prototype","non_intended_use":["clinical diagnosis","autonomous treatment"],"heldout_metrics":{"macro_f1":.7},"uncertainty_or_abstention":{"review_rate":.2},"limitations":["small cohort","domain shift","label uncertainty"],"human_oversight":"expert review required"}
    assert not validate(good)
    bad=dict(good); bad["test_groups"]=["p1"]
    assert validate(bad)
    print("Healthcare evidence gate self-test passed.")


def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("evidence",type=Path,nargs="?"); p.add_argument("--self-test",action="store_true"); args=p.parse_args()
    if args.self_test: self_test(); return 0
    if args.evidence is None: p.error("evidence JSON is required")
    payload=json.loads(args.evidence.read_text(encoding="utf-8")); errors=validate(payload); report={"project":payload.get("project"),"status":"PASS" if not errors else "FAIL","errors":errors,"notice":"Portfolio evidence gate only; PASS does not establish clinical validity or safety."}; print(json.dumps(report,indent=2)); return 1 if errors else 0

if __name__=="__main__": raise SystemExit(main())
