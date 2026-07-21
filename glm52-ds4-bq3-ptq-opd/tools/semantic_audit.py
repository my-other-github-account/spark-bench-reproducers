#!/usr/bin/env python3
"""Cross-file scientific and documentation consistency gate."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANONICAL_TRANSFER = "fa098459bdaeb09768900dd4663097a489c64e9410db46c2fc262d96151457cf"
STALE_TRANSFER = "fa098459bdaeb097f5dfe61a16c54112fe765ea816e4725113162cc4709db50a"
CODE_PARTIAL = "c63bf9f43aeef0f74306e4f66826ea53cbce98270276698bae97c442649354c0"
REQUIRED = (
    "README.md",
    "RESULTS.md",
    "CENSORING.md",
    "TRANSFER.md",
    "METHOD.md",
    "FAILURES.md",
    "NEXT.md",
    "REPRO.md",
)


def check_markdown_links(files: list[Path]) -> list[str]:
    failures: list[str] = []
    pattern = re.compile(r"(?<!!)\[[^]]*\]\(([^)]+)\)")
    for path in files:
        for target in pattern.findall(path.read_text()):
            target = target.strip().split()[0].strip("<>")
            if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            target = target.split("#", 1)[0]
            if target and not (path.parent / target).resolve().exists():
                failures.append(f"{path.name}: missing link target {target}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    checks: list[dict[str, object]] = []

    def add(name: str, passed: bool, detail: object) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    missing = [name for name in REQUIRED if not (ROOT / name).is_file()]
    add("required_documents", not missing, {"missing": missing})
    if missing:
        raise SystemExit("SEMANTIC_AUDIT_FAIL missing required documents")

    docs = {name: (ROOT / name).read_text() for name in REQUIRED}
    all_docs = "\n".join(docs.values())
    results = json.loads((ROOT / "receipts" / "SEALED_RESULTS.json").read_text())

    add(
        "sealed_objective",
        results.get("method", {}).get("objective") == "jsd",
        {"observed": results.get("method", {}).get("objective"), "expected": "jsd"},
    )
    forbidden_opkl = [
        phrase
        for phrase in (
            "OPKL is a static",
            "static OPKL",
            "teacher-forced OPKL",
            "OPKL is the static",
        )
        if phrase in all_docs
    ]
    add(
        "opkl_definition",
        "student-generated trajectories" in docs["README.md"]
        and "student-generated trajectory" in docs["METHOD.md"]
        and not forbidden_opkl,
        {"forbidden_phrases": forbidden_opkl},
    )
    transfer_index = results.get("transfer_step8", {})
    add(
        "authoritative_transfer8",
        CANONICAL_TRANSFER in docs["TRANSFER.md"]
        and transfer_index.get("sealed_verdict_receipt_sha256") == CANONICAL_TRANSFER
        and "NO_DECREASE_CLAIM" in docs["README.md"]
        and "NO_DECREASE_CLAIM" in docs["TRANSFER.md"]
        and "NO_DECREASE_CLAIM" in docs["NEXT.md"],
        {"indexed": transfer_index.get("sealed_verdict_receipt_sha256")},
    )
    add(
        "stale_transfer8_absent",
        STALE_TRANSFER not in all_docs
        and STALE_TRANSFER not in (ROOT / "receipts" / "SEALED_RESULTS.json").read_text(),
        {"stale": STALE_TRANSFER},
    )
    partial_index = results.get("transfer_step8_static_code_partial", {})
    add(
        "code_partial_scoped",
        CODE_PARTIAL in docs["TRANSFER.md"]
        and partial_index.get("source_receipt_sha256") == CODE_PARTIAL
        and "complete code-class partial" in docs["TRANSFER.md"]
        and "full 512-window cross-class" in docs["TRANSFER.md"]
        and "OPEN" in docs["TRANSFER.md"],
        {"indexed": partial_index.get("source_receipt_sha256")},
    )
    add(
        "trained146_heldout18_caveat",
        "146-task" in docs["README.md"]
        and "held-out 18 did not move" in docs["README.md"]
        and "Train/held-out boundary" in docs["RESULTS.md"],
        {},
    )
    add(
        "censoring_not_attractor",
        "right-censored" in docs["CENSORING.md"]
        and "not attractors" in docs["CENSORING.md"].lower()
        and "lower bound" in docs["CENSORING.md"],
        {},
    )
    add(
        "two_way_dissociation",
        "dissociation in both directions" in docs["TRANSFER.md"]
        and "Behavior can improve without static KLD improving" in docs["TRANSFER.md"]
        and "Static KLD can improve without behavior improving" in docs["TRANSFER.md"],
        {},
    )
    ownership_bad = re.findall(
        r"(?i)\b(?:our|ours|repaired)\b[^\n]{0,40}\b(?:IQ\w*|UD-[\w-]+)\b|"
        r"\b(?:IQ\w*|UD-[\w-]+)\b[^\n]{0,40}\b(?:our|ours|repaired)\b",
        all_docs,
    )
    add(
        "naming_and_ownership",
        not ownership_bad
        and "BQ3" in all_docs
        and "PTQ-OPD" in all_docs
        and "banana_bae" in all_docs
        and "Unsloth" in (ROOT / "receipts" / "SEALED_RESULTS.json").read_text(),
        {"bad_matches": ownership_bad[:10]},
    )

    link_failures = check_markdown_links([ROOT / name for name in REQUIRED])
    add("markdown_links", not link_failures, {"failures": link_failures})

    receipt_proc = subprocess.run(
        ["python3", str(ROOT / "tools" / "verify_receipts.py")],
        text=True,
        capture_output=True,
        check=False,
    )
    add(
        "receipt_bundle",
        receipt_proc.returncode == 0,
        {"stdout": receipt_proc.stdout.strip(), "stderr": receipt_proc.stderr.strip()},
    )

    failures = [check for check in checks if not check["passed"]]
    output = {
        "schema": "ptq-opd-semantic-audit-v1",
        "checks": checks,
        "summary": {
            "checks": len(checks),
            "passed": len(checks) - len(failures),
            "failed": len(failures),
            "verdict": "PASS" if not failures else "HOLD",
        },
    }
    if args.output:
        args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], sort_keys=True))
    if failures:
        for check in failures:
            print(f"FAIL {check['name']}: {check['detail']}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
