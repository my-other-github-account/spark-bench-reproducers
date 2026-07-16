#!/usr/bin/env python3
"""Validate the checked-in campaign publication without model weights."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "LEARNINGS.md",
    "RESUME.md",
    "REPRO.md",
    "NEXT_STEPS.md",
    "ladder/ANCHOR_TABLE.md",
    "repair/SEALED_REPAIR_REPLICATION.json",
    "repair/PROBE_TABLES.json",
    "repair/TRAJECTORIES.md",
    "repair/external-gate/README.md",
    "repair/rail512/README.md",
    "research-track/vq-gptq/PILOT_SUMMARY.json",
    "serving/WIRE_GATE_win0.json",
    "tooling/README.md",
    "tooling/export_arm4.py",
    "tooling/agg_rail.py",
    "tooling/fix_runpilot_env.py",
    "tooling/rail512_shard.sh",
    "environments/spark-1-repair.txt",
    "environments/spark-3-build.txt",
    "environments/spark-5-solve-repair.txt",
    "environments/spark-8-eval-serve.txt",
]
LINK = re.compile(r"(?<!!)\[[^]]*\]\(([^)]+)\)")


def fail(message: str) -> None:
    print(f"ERROR {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    for rel in REQUIRED:
        if not (ROOT / rel).is_file():
            fail(f"missing required artifact: {rel}")

    json_count = 0
    jsonl_rows = 0
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix == ".json":
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                fail(f"invalid JSON {path.relative_to(ROOT)}: {exc}")
            json_count += 1
        elif path.suffix == ".jsonl":
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    json.loads(line)
                except Exception as exc:
                    fail(f"invalid JSONL {path.relative_to(ROOT)}:{number}: {exc}")
                jsonl_rows += 1
        elif path.suffix == ".md":
            text = path.read_text(encoding="utf-8")
            for target in LINK.findall(text):
                target = target.split("#", 1)[0].strip()
                if not target or target.startswith(("http://", "https://", "mailto:", "$")):
                    continue
                if any(char in target for char in "*{}"):
                    continue
                candidate = (path.parent / target).resolve()
                if ROOT not in candidate.parents and candidate != ROOT:
                    fail(f"link escapes publication root: {path.relative_to(ROOT)} -> {target}")
                if not candidate.exists():
                    fail(f"broken link: {path.relative_to(ROOT)} -> {target}")

    seal = json.loads((ROOT / "repair/SEALED_REPAIR_REPLICATION.json").read_text())
    verdict = seal.get("verdict") or seal.get("replication", {}).get("verdict")
    if verdict != "REPLICATED_ABOVE_FLOOR_HELDOUT_KLD_REPAIR_ZERO_LEAKAGE":
        fail(f"unexpected formal repair verdict: {verdict!r}")

    print(f"PUBLICATION_VALIDATE_PASS json={json_count} jsonl_rows={jsonl_rows}")


if __name__ == "__main__":
    main()
