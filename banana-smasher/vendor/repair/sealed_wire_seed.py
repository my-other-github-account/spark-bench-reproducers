#!/usr/bin/env python3
"""Build a repair seed from exact cells already sealed in one wire inventory.

This is the privacy-clean, model-agnostic form of the corrected terminal-seed
logic: sources are joined by full cell identity and payload SHA, never guessed
from an earlier model tree or an ambient checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Iterable, List, Mapping


ALLOWED_TIERS = {"qtip2", "qtip3"}


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def build_seed(
    inventory_rows: Iterable[Mapping[str, Any]],
    updates: Iterable[Mapping[str, Any]],
    expected_updates: int = 24,
) -> List[Dict[str, Any]]:
    inventory: Dict[str, Mapping[str, Any]] = {}
    for row in inventory_rows:
        cell = row.get("cell")
        tier = row.get("tier")
        payload_sha = row.get("payload_sha256")
        if not isinstance(cell, str) or not cell:
            raise ValueError("inventory cell identity missing")
        if cell in inventory:
            raise ValueError("duplicate inventory cell: " + cell)
        if tier not in ALLOWED_TIERS:
            raise ValueError("unsupported inventory tier for {}: {}".format(cell, tier))
        if not _is_sha(payload_sha):
            raise ValueError("invalid payload SHA for " + cell)
        inventory[cell] = row

    requested = list(updates)
    if len(requested) != expected_updates:
        raise ValueError("expected {} updates, observed {}".format(expected_updates, len(requested)))
    targets = [row.get("target") for row in requested]
    if len(set(targets)) != len(targets):
        raise ValueError("duplicate target in update plan")

    result: List[Dict[str, Any]] = []
    for ordinal, update in enumerate(requested):
        target = update.get("target")
        source = update.get("source")
        if target not in inventory:
            raise ValueError("target absent from sealed-wire inventory: {}".format(target))
        if source not in inventory:
            raise ValueError("source absent from sealed-wire inventory: {}".format(source))
        target_row = inventory[target]
        source_row = inventory[source]
        if target_row["tier"] != source_row["tier"]:
            raise ValueError("tier mismatch for target {} and source {}".format(target, source))
        row = {
            "ordinal": ordinal,
            "target": target,
            "target_tier": target_row["tier"],
            "target_payload_sha256": target_row["payload_sha256"],
            "source": source,
            "source_tier": source_row["tier"],
            "source_payload_sha256": source_row["payload_sha256"],
            "source_law": "exact-sealed-wire-inventory-reuse",
        }
        row["row_sha256"] = hashlib.sha256(canonical_bytes(row)).hexdigest()
        result.append(row)
    return result


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name + ".tmp-", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--updates", type=Path, required=True)
    parser.add_argument("--expected-updates", type=int, default=24)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text())
    updates = json.loads(args.updates.read_text())
    seed = build_seed(inventory, updates, expected_updates=args.expected_updates)
    document = {
        "schema": "banana-smasher-sealed-wire-seed-v1",
        "status": "PASS",
        "updates": seed,
        "aggregate_sha256": hashlib.sha256(canonical_bytes(seed)).hexdigest(),
    }
    atomic_json(args.output, document)
    print(json.dumps({"status": "PASS", "updates": len(seed), "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
