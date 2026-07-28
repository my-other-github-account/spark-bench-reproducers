#!/usr/bin/env python3
"""Offline Docker build-context and copied-runtime closure check."""
from __future__ import annotations

import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    dockerfile = (ROOT / "Dockerfile").read_text()
    failures = []
    copies = []
    mappings = []
    for line_number, line in enumerate(dockerfile.splitlines(), 1):
        match = re.match(r"\s*COPY\s+([^\s]+)\s+([^\s]+)\s*$", line)
        if not match:
            continue
        source, destination = match.groups()
        copies.append(source)
        mappings.append((source, destination.rstrip("/")))
        if source.startswith(("/", "~")) or ".." + "/" in source:
            failures.append({"line": line_number, "source": source, "reason": "context escape"})
        elif not (ROOT / source).exists():
            failures.append({"line": line_number, "source": source, "reason": "missing"})

    required = {"vendor/runtime", "vendor/kernel", "configs", "locks"}
    missing = sorted(required - set(copies))
    if missing:
        failures.append({"missing_copy_roots": missing})

    freeze = json.loads((ROOT / "configs" / "RUNTIME_FREEZE.json").read_text())
    base = freeze["base_image"]
    pinned_base = "{}@{}".format(base["reference"], base["manifest_digest"])
    if pinned_base not in dockerfile:
        failures.append({"reason": "Docker base image is not pinned to RUNTIME_FREEZE digest"})

    lock_path = ROOT / "locks" / "requirements-runtime.txt"
    requirements = {}
    for line in lock_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.count("==") != 1:
            failures.append({"requirement": line, "reason": "dependency is not exact-pinned"})
            continue
        name, version = line.split("==", 1)
        requirements[name] = version
    if "-r /opt/genesis/locks/requirements-runtime.txt" not in dockerfile:
        failures.append({"reason": "Docker build does not install the shipped runtime lock"})
    for name in ("flashinfer-python", "tokenizers", "transformers", "triton", "vllm"):
        expected_version = freeze["runtime"][name]
        if requirements.get(name) != expected_version:
            failures.append({
                "requirement": name,
                "expected": expected_version,
                "observed": requirements.get(name),
                "reason": "runtime lock disagrees with RUNTIME_FREEZE",
            })

    runtime_script = ROOT / "vendor" / "runtime" / "serve.sh"
    runtime_references = sorted(set(re.findall(r"/opt/genesis/[A-Za-z0-9._/-]+", runtime_script.read_text())))
    resolved_references = {}
    for reference in runtime_references:
        candidates = []
        for source, destination in mappings:
            if reference == destination:
                candidates.append(ROOT / source)
            elif reference.startswith(destination + "/"):
                suffix = reference[len(destination) + 1 :]
                candidates.append(ROOT / source / suffix)
        existing = [path for path in candidates if path.exists()]
        if not existing:
            failures.append({"reference": reference, "reason": "runtime path is not supplied by Docker COPY"})
        else:
            resolved_references[reference] = existing[0].relative_to(ROOT).as_posix()

    result = {
        "schema": "banana-smasher-docker-context-verification-v2",
        "status": "PASS" if not failures else "FAIL",
        "copies": copies,
        "base_image": pinned_base,
        "exact_pinned_requirements": len(requirements),
        "runtime_references": resolved_references,
        "failures": failures,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
