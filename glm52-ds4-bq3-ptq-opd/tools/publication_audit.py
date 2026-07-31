#!/usr/bin/env python3
"""Fail-closed privacy, provenance, and payload audit for this reproducer package."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
TEXT_SUFFIXES = {
    "",
    ".csv",
    ".json",
    ".md",
    ".patch",
    ".py",
    ".sha256",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
EXCLUDED_PUBLIC_PAYLOADS = {
    "docker/artifacts/tokenizer.json",
    "docker/provenance/pip-freeze.sanitized.txt",
}

# Construct sensitive literals so this audit does not flag its own policy table.
JOINED_FORBIDDEN = [
    "mac" + "mini",
    "d" + "nola",
    "Da" + "vid",
]
PATTERNS = {
    "absolute home path": re.compile(r"/(?:Users|home)/[^\s\"'`]+"),
    "private host": re.compile(r"\bspark-[0-9]+(?:\b|[-_])", re.IGNORECASE),
    "task identifier": re.compile(r"t_[0-9a-f]{8}", re.IGNORECASE),
    "legacy public identity": re.compile("banana_" + "baeee", re.IGNORECASE),
    "private IPv4 address": re.compile(
        r"(?<![0-9])(?:10\.(?:[0-9]{1,3}\.){2}[0-9]{1,3}|"
        r"192\.168\.(?:[0-9]{1,3}\.)[0-9]{1,3}|"
        r"172\.(?:1[6-9]|2[0-9]|3[01])\.(?:[0-9]{1,3}\.)[0-9]{1,3}|"
        r"100\.(?:6[4-9]|[7-9][0-9]|1[01][0-9]|12[0-7])\.(?:[0-9]{1,3}\.)[0-9]{1,3})(?![0-9])"
    ),
    "private mission path": re.compile(r"(?:^|[/\\])missions(?:[/\\]|$)", re.IGNORECASE),
    "BQ3 misnamed as IQ3": re.compile(r"IQ3_BIN|repaired-IQ3|IQ3 artifact", re.IGNORECASE),
}

TOOLS = ROOT / "tools/qtip2-backpack-campaign"
MANIFEST_PATH = TOOLS / "TOOLS_MANIFEST.json"
MANIFEST_MD_PATH = TOOLS / "TOOLS_MANIFEST.md"
NON_PAYLOAD_FILES = {"TOOLS_MANIFEST.json", "TOOLS_MANIFEST.md"}
FORBIDDEN_DATA_SUFFIXES = {
    ".bin",
    ".gguf",
    ".npy",
    ".npz",
    ".pt",
    ".pth",
    ".safetensors",
}
REQUIRED_PREFIXES = {
    "solver/p629/",
    "solver/p637/",
    "builders/wire/",
    "builders/qtip-rep16/",
    "builders/packers/",
    "rail/p671/",
    "dose/p600/",
    "dose/p613-acceleration/",
    "dose/p662-candidate/",
    "dose/p672-package/",
    "misc/kld/",
    "misc/qsfp/",
    "misc/teacher-sharding/",
}
REQUIRED_FILES = {
    "solver/p629/solve_global_ab.py",
    "solver/p637/solve_actual.py",
    "solver/p637/solve_actual_respend.py",
    "solver/p637/solve_lp_bound.py",
    "solver/INPUT_MANIFEST_SCHEMA.json",
    "builders/wire/canonical_shared_builder.py",
    "builders/wire/build_shard.py",
    "builders/wire/build_overlay_shard.py",
    "builders/wire/pilot_code/gptqv2_pilot.py",
    "builders/wire/pilot_code/vqw2_pilot.py",
    "builders/qtip-rep16/qtip_rate_unit_p541.py",
    "builders/packers/t8192_ds4_build_v3.py",
    "rail/banana_smasher_remote_full512.py",
    "rail/full512_safety.py",
    "rail/provenance/HARNESS_FIX_RETIRE_SCRATCH.json",
    "rail/p651_overlay_rail.py",
    "rail/p671/p671_slice_w064_127.py",
    "rail/p671/launch_p671_slice_w064_127.sh",
    "rail/p632_score.py",
    "dose/p600/banana_smasher_basic_repair.py",
    "dose/p600/run_dose2.sh",
    "dose/p613-acceleration/banana_smasher_basic_repair_accel.py",
    "dose/p662-candidate/banana_smasher_basic_repair.py",
    "dose/p672-package/verify_bundle.py",
    "misc/kld/score_p623.py",
    "misc/teacher-sharding/coordinate_merge_verify.py",
    "misc/teacher-sharding/memory_monitor.py",
    "misc/qsfp/p613_multistream_stage.py",
    "misc/KASA_RECOVERY.md",
}

CAMPAIGN_SYNC_FILES = {
    "CURRENT_BEST.md",
    "NEW_MODEL_CHECKLIST.md",
    "SERVE_RUNBOOK.md",
    "qtip2-backpack-campaign/UPDATE_2026-07-29.md",
    "qtip2-backpack-campaign/MEASUREMENT_INTEGRITY.md",
    "qtip2-backpack-campaign/BANANA_PACK_SPEC.md",
    "qtip2-backpack-campaign/PUBLICATION_STATUS_2026-07-29.json",
}
STATUS_PATH = ROOT / "qtip2-backpack-campaign/PUBLICATION_STATUS_2026-07-29.json"
TRUE_C_PINS = {
    "byte_envelope": 101346700411,
    "overlay_sha256": "9a4b709851c62c32f59b17556ef14d53e89cbbfc0fcc93686fc51530e4cf4d62",
    "pack_manifest_sha256": "3650fe7e627b180a979fb8304f90e888333671cf03334e965fd5b14b7393b220",
    "planes_manifest_sha256": "b524c5a67bbcad6aef14d70b464b46097302bf004bb75c1265f2ff683bae083d",
}
HOLDOUT_PINS = {
    "manifest_sha256": "063b7552deeda0494ef623b048a325e271671867df7501ffdc79faca6708fe1b",
    "windows_sha256": "2de3ac4110ade4efe7c1b9f1482ef920352142cbee549cffb475f0aa91cc7896",
    "disjointness_receipt_sha256": "79943e7398c665c88a223e5eb41f4958787d3cd10b1da845fe99b567f456492e",
    "seal_sha256": "9480ded58b214d09f3c71c000b79b9a72d3ce20b7c674f412bd933ce8c44f5d5",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_manifest() -> list[str]:
    failures: list[str] = []
    if not MANIFEST_PATH.is_file() or not MANIFEST_MD_PATH.is_file():
        return ["missing TOOLS_MANIFEST.json or TOOLS_MANIFEST.md"]
    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        return [f"invalid TOOLS_MANIFEST.json: {error}"]

    rows = manifest.get("files")
    if not isinstance(rows, list):
        return ["TOOLS_MANIFEST.json files must be an array"]
    by_path: dict[str, dict[str, object]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            failures.append(f"manifest row {index} is not an object")
            continue
        path_text = row.get("path")
        if not isinstance(path_text, str) or not path_text:
            failures.append(f"manifest row {index} has invalid path")
            continue
        if path_text in by_path:
            failures.append(f"duplicate manifest path: {path_text}")
            continue
        by_path[path_text] = row
        path = TOOLS / path_text
        if not path.is_file():
            failures.append(f"manifest path missing: {path_text}")
            continue
        actual = sha256(path)
        if row.get("shipped_sha256") != actual:
            failures.append(f"shipped SHA mismatch: {path_text}")
        for required in ("repro_stage", "source_sha256", "origin", "source_verification"):
            if required not in row:
                failures.append(f"manifest field missing for {path_text}: {required}")
        source_hash = row.get("source_sha256")
        if not isinstance(source_hash, str) or re.fullmatch(r"[0-9a-f]{64}", source_hash) is None:
            failures.append(f"invalid source SHA-256: {path_text}")

    payload_paths = {
        path.relative_to(TOOLS).as_posix()
        for path in TOOLS.rglob("*")
        if path.is_file()
        and path.relative_to(TOOLS).as_posix() not in NON_PAYLOAD_FILES
        and "__pycache__" not in path.parts
    }
    listed_paths = set(by_path)
    for path_text in sorted(payload_paths - listed_paths):
        failures.append(f"unlisted tools payload: {path_text}")
    for path_text in sorted(listed_paths - payload_paths):
        failures.append(f"manifest lists non-payload path: {path_text}")

    for prefix in sorted(REQUIRED_PREFIXES):
        if not any(path.startswith(prefix) for path in listed_paths):
            failures.append(f"required reproduction family absent: {prefix}")
    for path_text in sorted(REQUIRED_FILES - listed_paths):
        failures.append(f"required reproduction file absent: {path_text}")
    builder = by_path.get("builders/wire/canonical_shared_builder.py", {})
    if builder.get("source_sha256") != "60b594ac38e4973eaaecb76c708b555418406eb697414d2563aeb1e978268a7e":
        failures.append("canonical_shared_builder.py source pin is not 60b594ac…")

    manifest_md = MANIFEST_MD_PATH.read_text(encoding="utf-8")
    for path_text in sorted(listed_paths):
        if f"`{path_text}`" not in manifest_md:
            failures.append(f"TOOLS_MANIFEST.md missing path: {path_text}")
    for path in TOOLS.rglob("*"):
        if path.is_file() and path.suffix.lower() in FORBIDDEN_DATA_SUFFIXES:
            failures.append(f"forbidden data payload bundled: {path.relative_to(TOOLS)}")
    return failures


def audit_campaign_sync() -> list[str]:
    """Bind the Jul-29 narrative to machine-readable pins and validity labels."""
    failures: list[str] = []
    for relative in sorted(CAMPAIGN_SYNC_FILES):
        if not (ROOT / relative).is_file():
            failures.append(f"campaign sync file absent: {relative}")
    if failures or not STATUS_PATH.is_file():
        return failures

    try:
        status = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        return [f"invalid PUBLICATION_STATUS_2026-07-29.json: {error}"]

    if status.get("schema") != "banana_bae.publication_status.v1":
        failures.append("unexpected campaign publication-status schema")
    if status.get("public_operator") != "banana_bae":
        failures.append("campaign publication operator is not banana_bae")
    artifact = status.get("artifact", {})
    for field, expected in TRUE_C_PINS.items():
        if artifact.get(field) != expected:
            failures.append(f"TRUE-C status pin mismatch: {field}")

    quality = status.get("quality", {})
    if quality.get("global_kld") != 0.06484517121688964:
        failures.append("U004 BALANCED64 global KLD mismatch")
    if quality.get("validity") != "MEASURED_DEVELOPMENT_READ__DESIGN_AND_SELECTION_LEAKAGE":
        failures.append("U004 validity does not disclose design/selection leakage")
    if quality.get("final_holdout_score") is not None:
        failures.append("unsealed HOLDOUT512 score must remain null")

    serving = status.get("serving", {})
    if serving.get("validity") != "MEASURED_PRODUCT_PASS":
        failures.append("serving result is not labeled MEASURED_PRODUCT_PASS")
    if serving.get("receipt_sha256") != "3117274cf826804437509475a2294ea773d9ee5e64723df9f657c0123c28a413":
        failures.append("serving receipt SHA mismatch")
    if serving.get("decode256_tok_s") != [15.897619, 15.902757, 16.044741, 15.928163, 15.969125]:
        failures.append("serving five-row decode vector mismatch")
    for field, expected in {
        "exact_2048_prefill_tok_s": 864.804416,
        "warm_ttft_s": 2.368165,
        "exact_8192_prefill_tok_s": 840.413577,
        "vm_swap_bytes": 0,
        "gates_passed": 18,
        "gates_total": 18,
    }.items():
        if serving.get(field) != expected:
            failures.append(f"serving status mismatch: {field}")

    concurrency = status.get("concurrency", {})
    if concurrency.get("validity") != "MEASURED_NO_GO__NO_VALID_C1_C2_C4_C8_MATRIX":
        failures.append("concurrency must remain labeled measured NO-GO")
    for field, expected in {
        "c1_aggregate_tok_s": 15.351984064,
        "c1_exact_2048_prefill_tok_s": 838.865555757,
        "c2_aggregate_tok_s": 2.845162,
        "c2_over_c1_ratio": 0.185329,
        "c4_c8_release_gate": "same-method C2 > 1.2x C1",
    }.items():
        if concurrency.get(field) != expected:
            failures.append(f"concurrency status mismatch: {field}")
    if concurrency.get("c4_aggregate_tok_s") is not None or concurrency.get("c8_aggregate_tok_s") is not None:
        failures.append("unmeasured C4/C8 cells must remain null")

    comparators = status.get("comparators", {})
    for name in ("UD-IQ4_XS", "UD-IQ3_XXS"):
        row = comparators.get(name, {})
        for field in ("exact_common_byte_accounting", "holdout512_v1", "same_method_serve"):
            if row.get(field) != "TBD":
                failures.append(f"{name} pending comparator cell must be TBD: {field}")

    holdout = status.get("holdout", {})
    for field, expected in HOLDOUT_PINS.items():
        if holdout.get(field) != expected:
            failures.append(f"HOLDOUT512 status pin mismatch: {field}")
    if holdout.get("windows") != 512 or holdout.get("tokens_per_window") != 1024:
        failures.append("HOLDOUT512 shape mismatch")
    if holdout.get("seed") != 20260730:
        failures.append("HOLDOUT512 seed mismatch")
    if holdout.get("validity") != "ADOPTED_STANDING_SCORING_ONLY_ASSET":
        failures.append("HOLDOUT512 standing-asset validity missing")

    required_markers = {
        "CURRENT_BEST.md": ["MEASURED NO-GO", "HOLDOUT512_V1", "18 product gates"],
        "NEW_MODEL_CHECKLIST.md": ["Check canon before claiming absence", "wire is not done until its serving-format export", "planes13", "TBD"],
        "SERVE_RUNBOOK.md": ["usage.completion_tokens", "KeyError: planes13", "No valid actual-Wire-C monotonic ladder"],
        "qtip2-backpack-campaign/UPDATE_2026-07-29.md": ["Grand table", "strict-C1 HumanEval", "FLAG_DESIGN_TIME_AND_SELECTION_LEAKAGE", "0.185329", "TBD"],
        "qtip2-backpack-campaign/MEASUREMENT_INTEGRITY.md": ["scoring-only forever", "bank itself remains private", "no full HOLDOUT512_V1 quality score"],
        "qtip2-backpack-campaign/BANANA_PACK_SPEC.md": ["BANANA_PACK_SPEC v1", "BananaSmasher instance", "`smash`", "BANANA-2.75", "BANANA-2.50", "BANANA-2.25", "BANANA-2.00"],
    }
    for relative, markers in required_markers.items():
        text = (ROOT / relative).read_text(encoding="utf-8")
        for marker in markers:
            if marker not in text:
                failures.append(f"{relative} missing campaign marker: {marker}")
    return failures


def text_files() -> list[Path]:
    files = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.resolve() == SELF:
            continue
        if any(part in {".git", "__pycache__", ".venv"} for part in path.parts):
            continue
        relative = path.relative_to(ROOT).as_posix()
        if relative in EXCLUDED_PUBLIC_PAYLOADS:
            continue
        if path.suffix.lower() in TEXT_SUFFIXES:
            files.append(path)
    return sorted(files)


def main() -> None:
    failures = audit_manifest() + audit_campaign_sync()
    files = text_files()
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        relative = path.relative_to(ROOT).as_posix()
        for line_number, line in enumerate(text.splitlines(), 1):
            for label, pattern in PATTERNS.items():
                if pattern.search(line):
                    failures.append(f"{relative}:{line_number}: {label}")
            for forbidden in JOINED_FORBIDDEN:
                if forbidden.lower() in line.lower():
                    failures.append(f"{relative}:{line_number}: forbidden identity")
    if failures:
        print("PUBLICATION_AUDIT_FAIL")
        print("\n".join(failures))
        raise SystemExit(1)
    print(f"PUBLICATION_AUDIT_PASS files={len(files)} manifest_rows={len(json.loads(MANIFEST_PATH.read_text())['files'])}")


if __name__ == "__main__":
    main()
