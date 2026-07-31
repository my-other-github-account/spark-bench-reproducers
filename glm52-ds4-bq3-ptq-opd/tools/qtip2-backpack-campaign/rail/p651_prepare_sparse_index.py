#!/usr/bin/env python3
"""Prepare the fail-closed P651 sparse-overlay resolver index from final P640/P647 receipts."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

TASK = "PUBLIC_TASK"
MISSION = Path("$HOME/run-bundles/P651_STREAM_CONSUMER_PUBLIC_TASK_s7")
META = MISSION / "inputs/P640_FINAL_META_RESPENT"
OUTPUT = MISSION / "inputs/SPARSE_OVERLAY_INDEX.json"
ASSIGNMENT_SHA = "c030883fddb1217529d67444d08257c4a1df18e2adbc93be092aba3d3611bc65"
MAP_SHA = "36d0841986d5781186f766b3815e4b3c6332eece2090d3e6d73e7e3ffa33dc07"
BASE_ASSIGNMENT_SHA = "c9fb72e2bf7416ef48f33df229f9a3b5b5dd4f9e9b35a610d83fb1c49f4a050d"
BUILD_PLAN_SHA = "c1ed36565f7d89fbb4c8a6f477f871425e72360fcbd635bd7a1118d5d86c19fe"
QTIP_EXPECTED_SHA = "c90558fc2095affccdb6b0d86b79bfe594bdaccd3a2cb886d1b4ece40ad7a0ff"
BUILDER_SHA = "60b594ac38e4973eaaecb76c708b555418406eb697414d2563aeb1e978268a7e"
WIRE_SHA = "c24a1c0568a00fcb8460d7edfb7630187ef10c98e9d0c25c87aa0bccb1d89755"
WIRE_BYTES = 101346521679
P632_SCORER_SHA = "5c16e62c32e6936223c54e2b3cf9394a1d0f87833cc409360e82e0341954c12f"
SOURCE_HOST = "203.0.113.6"
SOURCE_ROOT = Path("$HOME/run-bundles/P640_BANANA_SMASHER_QTIP2_WIRE_PUBLIC_TASK_s6/WIRE_STREAM_IN/P647_RESPENT")

MANIFESTS = [
    {
        "name": "L00_11",
        "path": META / "WIRE_STREAM_IN/P647_RESPENT/L00_11/OVERLAY_PUBLIC_TASK/receipts/SHARD_MANIFEST.json",
        "sha256": "20051b54730d9e46237c5cc84a94c6c16f24896f157293d758a83e49742568ea",
        "source_root": SOURCE_ROOT / "L00_11/OVERLAY_PUBLIC_TASK",
        "rows_key": "rows",
    },
    {
        "name": "L12_22",
        "path": META / "WIRE_STREAM_IN/P647_RESPENT/L12_22/OVERLAY/receipts/SHARD_MANIFEST.json",
        "sha256": "7262403f9fc5567e77026ebb0bb9f9ead660c82b3d19ebc702296a93e878e2db",
        "source_root": SOURCE_ROOT / "L12_22/OVERLAY",
        "rows_key": "rows",
    },
    {
        "name": "L23_32",
        "path": META / "WIRE_STREAM_IN/P647_RESPENT/L23_32/overlay/receipts/SHARD_MANIFEST.json",
        "sha256": "a60e2efea128516a2148b747128f69d78a5ef18c3a62a09cf67503c1f1c6b3d5",
        "source_root": SOURCE_ROOT / "L23_32/overlay",
        "rows_key": "rows",
    },
    {
        "name": "L33_42",
        "path": META / "WIRE_STREAM_IN/P647_RESPENT/L33_42/P647_RESPENT_OVERLAY_L33_42_FINAL.json",
        "sha256": "1abbcf65837c573b365216f595cf87bca9658e70c9c23998740139b4825e95c8",
        "source_root": None,
        "rows_key": "changed_artifact_validation_rows",
    },
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def exact_claim() -> tuple[bytes, dict[str, Any]]:
    path = Path("$HOME/HOST_CLAIM.json")
    raw = path.read_bytes()
    obj = json.loads(raw)
    exact = {"host": "compute-node-7", "owner": TASK, "task_id": TASK, "mission": str(MISSION)}
    drift = {k: (obj.get(k), v) for k, v in exact.items() if obj.get(k) != v}
    if drift:
        raise RuntimeError(f"claim drift: {drift}")
    return raw, obj


def flatten(doc: dict[str, Any]) -> dict[tuple[int, int, str], str]:
    result: dict[tuple[int, int, str], str] = {}
    for layer, experts in doc["assignment"].items():
        for expert, projections in experts.items():
            for projection, tier in projections.items():
                result[(int(layer), int(expert), str(projection))] = str(tier)
    return result


def checked_meta(path: Path, expected_sha: str) -> dict[str, Any]:
    if not path.is_file() or sha256(path) != expected_sha:
        raise RuntimeError(f"metadata SHA drift: {path}")
    raw = path.read_text()
    if "26d0cd3b" in raw or "VQ3_K4096" in raw or "raw VQ3" in raw:
        raise RuntimeError(f"superseded artifact marker present: {path}")
    return json.loads(raw)


def main() -> int:
    if OUTPUT.exists():
        raise RuntimeError(f"once-only sparse index exists: {OUTPUT}")
    claim_raw, _ = exact_claim()
    assignment_path = META / "inputs/ASSIGNMENT_RESPENT.json"
    base_path = META / "inputs/CURRENT_BANANA_SMASHER_ASSIGNMENT.json"
    plan_path = META / "inputs/BUILD_PLAN.json"
    qtip_path = META / "inputs/QTIP_SELECTED_EXPECTED.json"
    if sha256(assignment_path) != ASSIGNMENT_SHA or sha256(base_path) != BASE_ASSIGNMENT_SHA:
        raise RuntimeError("assignment pin mismatch")
    if sha256(plan_path) != BUILD_PLAN_SHA or sha256(qtip_path) != QTIP_EXPECTED_SHA:
        raise RuntimeError("plan/QTIP expectation pin mismatch")
    final_doc = json.loads(assignment_path.read_text())
    base_doc = json.loads(base_path.read_text())
    plan = json.loads(plan_path.read_text())
    qtip_doc = json.loads(qtip_path.read_text())
    if final_doc.get("assignment_map_sha256") != MAP_SHA:
        raise RuntimeError("embedded final assignment map SHA drift")
    if plan.get("authoritative_assignment_map_sha256") != MAP_SHA or int(plan.get("authoritative_exact_wire_bytes", -1)) != WIRE_BYTES:
        raise RuntimeError("build plan authority drift")
    final = flatten(final_doc)
    base = flatten(base_doc)
    if set(final) != set(base) or len(final) != 43 * 256 * 2:
        raise RuntimeError("assignment universe drift")
    expected_diff = {
        key: {"old": base[key], "new": final[key]}
        for key in final if final[key] != base[key]
    }
    if len(expected_diff) != 1411:
        raise RuntimeError(f"assignment diff count drift: {len(expected_diff)}")
    plan_diff = {
        (int(row["layer"]), int(row["expert"]), str(row["projection"])): {"old": str(row["old"]), "new": str(row["new"])}
        for row in plan["rows"]
    }
    if plan_diff != expected_diff:
        raise RuntimeError("BUILD_PLAN rows do not exactly equal assignment diff")
    qtip_expected = {
        (int(row["layer"]), int(row["expert"]), str(row["projection"])): row
        for row in qtip_doc["rows"]
    }
    if len(qtip_expected) != 406 or qtip_doc.get("assignment_sha256") != ASSIGNMENT_SHA:
        raise RuntimeError("QTIP exact REP-16 expectation surface drift")

    rows: list[dict[str, Any]] = []
    manifest_receipts = []
    for spec in MANIFESTS:
        path = Path(spec["path"])
        doc = checked_meta(path, str(spec["sha256"]))
        if doc.get("status") != "PASS":
            raise RuntimeError(f"manifest status drift: {path}")
        manifest_assignment = doc.get("assignment_sha256") or doc.get("assignment_file_sha256")
        manifest_map = doc.get("assignment_map_sha256") or doc.get("authoritative_assignment_map_sha256")
        manifest_wire = doc.get("expected_final_wire_bytes") or doc.get("authoritative_exact_wire_bytes")
        if manifest_assignment != ASSIGNMENT_SHA or manifest_map != MAP_SHA or int(manifest_wire or -1) != WIRE_BYTES:
            raise RuntimeError(f"manifest authority drift: {path}")
        source_rows = doc[str(spec["rows_key"])]
        manifest_receipts.append({"name": spec["name"], "path": str(path), "sha256": spec["sha256"], "rows": len(source_rows)})
        for source in source_rows:
            if spec["name"] == "L33_42":
                ident = source["identity"]
                layer, expert, projection = int(ident[0]), int(ident[1]), str(ident[2])
                source_root = SOURCE_ROOT / f"L33_42/L{layer:03d}/overlay"
                kind = str(source["kind"])
                old, new = str(source["old"]), str(source["new"])
            else:
                layer, expert, projection = int(source["layer"]), int(source["expert"]), str(source["projection"])
                source_root = Path(str(spec["source_root"]))
                kind = str(source["kind"])
                old, new = str(source["old"]), str(source["new"])
                if source.get("assignment_sha256") != ASSIGNMENT_SHA or source.get("assignment_map_sha256") != MAP_SHA:
                    raise RuntimeError(f"row authority drift: {(layer, expert, projection)}")
            key = (layer, expert, projection)
            if expected_diff.get(key) != {"old": old, "new": new}:
                raise RuntimeError(f"row assignment mismatch: {key}")
            artifact_name = Path(str(source["artifact"])).name
            artifact = {
                "path": f"cells/{artifact_name}",
                "bytes": int(source["artifact_bytes"]),
                "sha256": str(source["artifact_sha256"]),
            }
            row: dict[str, Any] = {
                "layer": layer, "expert": expert, "projection": projection,
                "old": old, "new": new, "kind": kind,
                "source_root": str(source_root), "artifact": artifact,
            }
            if kind == "banana_smasher_vq_rebuilt_cell":
                try:
                    d_text, k_text = new.split("_k", 1)
                    d, k = int(d_text[1:]), int(k_text)
                except Exception as exc:
                    raise RuntimeError(f"invalid VQ tier {new}: {key}") from exc
                if source.get("d") not in (None, d) or source.get("k") not in (None, k) or source.get("fp16_codebook_replay_exact") is not True:
                    raise RuntimeError(f"VQ replay/tier drift: {key}")
                codebook_name = Path(str(source["codebook"])).name
                row.update({
                    "d": d, "k": k,
                    "codebook": {
                        "path": f"codebooks/{codebook_name}",
                        "bytes": d * k * 2,
                        "sha256": str(source["codebook_sha256"]),
                    },
                    "canonical_builder_sha256": str(source.get("canonical_builder_sha256") or BUILDER_SHA),
                })
                if row["canonical_builder_sha256"] != BUILDER_SHA:
                    raise RuntimeError(f"canonical VQ builder drift: {key}")
            elif kind == "qtip2_exact_copy":
                expected = qtip_expected.get(key)
                if expected is None or new != "qtip2_2.0117":
                    raise RuntimeError(f"QTIP assignment/expectation missing: {key}")
                if (
                    artifact["bytes"] != int(expected["artifact_bytes"])
                    or artifact["sha256"] != expected["artifact_sha256"]
                    or source.get("source_artifact_sha256") != artifact["sha256"]
                ):
                    raise RuntimeError(f"QTIP exact REP-16 byte selection drift: {key}")
                row.update({
                    "source_manifest": str(source["source_manifest"]),
                    "logical_wire_bytes": int(source["logical_wire_bytes"]),
                    "logical_bpw": float(source["logical_bpw"]),
                    "qtip_expected_basename": str(expected["basename"]),
                })
            else:
                raise RuntimeError(f"unsupported/superseded changed-cell kind {kind}: {key}")
            rows.append(row)

    identities = [(r["layer"], r["expert"], r["projection"]) for r in rows]
    if len(rows) != 1411 or len(set(identities)) != 1411 or set(identities) != set(expected_diff):
        raise RuntimeError("sparse row exact coverage/uniqueness drift")
    if Counter(r["kind"] for r in rows) != Counter({"banana_smasher_vq_rebuilt_cell": 1005, "qtip2_exact_copy": 406}):
        raise RuntimeError("sparse row kind counts drift")

    by_layer: dict[str, Any] = {}
    for layer in range(43):
        layer_rows = sorted((r for r in rows if r["layer"] == layer), key=lambda r: (r["expert"], r["projection"]))
        files: dict[tuple[str, str], dict[str, Any]] = {}
        roots = {r["source_root"] for r in layer_rows}
        if len(roots) > 1:
            raise RuntimeError(f"multiple payload roots within layer {layer}: {roots}")
        for row in layer_rows:
            for key in ("artifact", "codebook"):
                item = row.get(key)
                if not item:
                    continue
                ident = (row["source_root"], item["path"])
                file_row = {"source_root": row["source_root"], **item, "role": key}
                prior = files.setdefault(ident, file_row)
                if {k: prior[k] for k in ("bytes", "sha256")} != {k: file_row[k] for k in ("bytes", "sha256")}:
                    raise RuntimeError(f"duplicate file identity disagreement: {ident}")
        by_layer[str(layer)] = {
            "layer": layer,
            "changed_cells": len(layer_rows),
            "vq_cells": sum(r["kind"] == "banana_smasher_vq_rebuilt_cell" for r in layer_rows),
            "qtip2_cells": sum(r["kind"] == "qtip2_exact_copy" for r in layer_rows),
            "unchanged_copythrough_cells": 512 - len(layer_rows),
            "source_root": next(iter(roots)) if roots else None,
            "files": sorted(files.values(), key=lambda x: x["path"]),
            "payload_bytes": sum(int(x["bytes"]) for x in files.values()),
            "rows": layer_rows,
        }
    if sum(v["changed_cells"] for v in by_layer.values()) != 1411:
        raise RuntimeError("layer row count drift")
    if sum(v["unchanged_copythrough_cells"] for v in by_layer.values()) != 43 * 512 - 1411:
        raise RuntimeError("unchanged copy-through count drift")

    payload = {
        "schema": "p651-p640-sparse-overlay-resolver-index-v1",
        "status": "PASS_FAIL_CLOSED_FINAL_RESPENT_ONLY",
        "task_id": TASK,
        "host": "compute-node-7",
        "source_task_id": "PUBLIC_TASK",
        "source_host": "compute-node-6",
        "source_qsfp": SOURCE_HOST,
        "assignment_file_sha256": ASSIGNMENT_SHA,
        "assignment_map_sha256": MAP_SHA,
        "base_assignment_sha256": BASE_ASSIGNMENT_SHA,
        "base_wire_manifest_sha256": WIRE_SHA,
        "exact_wire_bytes": WIRE_BYTES,
        "build_plan_sha256": BUILD_PLAN_SHA,
        "qtip_selected_expected_sha256": QTIP_EXPECTED_SHA,
        "canonical_shared_builder_sha256": BUILDER_SHA,
        "pinned_p632_scorer_sha256": P632_SCORER_SHA,
        "manifest_receipts": manifest_receipts,
        "row_count": len(rows),
        "row_set_sha256": canonical_sha(rows),
        "kind_counts": dict(Counter(r["kind"] for r in rows)),
        "layers": by_layer,
        "payload_file_count": sum(len(v["files"]) for v in by_layer.values()),
        "payload_bytes_per_single_rail_pass": sum(v["payload_bytes"] for v in by_layer.values()),
        "transfer_policy": "per-layer bounded scratch; >=4 parallel direct-QSFP rsync streams whenever layer has >=4 payload files; SHA-256 every received file; retire after GPU overlay",
        "base_policy": "sealed current-wire read-only direct local mmap; unchanged cells never rebuilt or transferred",
        "rejections": ["ASSIGNMENT_WITH map 26d0cd3b", "raw VQ3_K4096", "superseded full-tier shard outputs"],
        "claim_sha256": hashlib.sha256(claim_raw).hexdigest(),
        "created_unix": time.time(),
    }
    if exact_claim()[0] != claim_raw:
        raise RuntimeError("claim changed while preparing sparse resolver")
    atomic_json(OUTPUT, payload)
    print(json.dumps({
        "status": payload["status"], "rows": payload["row_count"],
        "files": payload["payload_file_count"],
        "payload_bytes_per_pass": payload["payload_bytes_per_single_rail_pass"],
        "index": str(OUTPUT), "index_sha256": sha256(OUTPUT),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
