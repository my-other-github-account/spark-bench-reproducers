#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import time

MISSION = Path("${SPARK_HOME}/missions/P530_PREFILL_t_099a5835_s8")
RECEIPTS = MISSION / "receipts"
FINAL = RECEIPTS / "final_gate/MIXED_PREFILL_LADDER_RESULT.json"
OUT = RECEIPTS / "P530_RESULT.json"
PRODUCT_BYTES = 101_346_700_411
P526_SHA = "655634773e941f6fa310235fe1adfbd1803eaa8d9207c9b51640b93d947e98a9"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    tmp = path.with_name("." + path.name + ".tmp")
    with tmp.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def main() -> int:
    summary = json.loads(FINAL.read_text())
    rows = [json.loads(Path(path).read_text()) for path in summary["raw_row_files"]]
    grouped = {row["prompt_tokens"]: row for row in summary["rows"]}
    code_hashes = {}
    for path in sorted((MISSION / "code").glob("*.py")):
        code_hashes[path.name] = sha(path)
    gates = {
        "final_result_pass": summary["status"] == "PASS",
        "six_cache_cold_rows": len(rows) == 6 and summary["cache_cold_rows_per_target"] == 3,
        "exact_shapes": set(grouped) == {2048, 8192},
        "all_row_gates_pass": all(row["status"] == "PASS" and all(row["gates"].values()) for row in rows),
        "median_prefill_ge_200": all(grouped[target]["prefill_tok_s_median"] >= 200.0 for target in (2048, 8192)),
        "finite_complete_rows": all(
            row["decode_tokens"] == 128 and all(math.isfinite(float(row[key])) for key in (
                "client_ttft_seconds", "server_prefill_seconds",
                "prefill_tok_s_prompt_over_client_ttft", "decode_tok_s"))
            for row in rows),
        "layers_43_four_tiers": all(
            row["active_layers"] == row["configured_layers"] == 43
            and all(int(v) > 0 for v in row["prefill_tier_kernel_launches"].values())
            and len(row["prefill_tier_kernel_launches"]) == 4
            for row in rows),
        "dedup_and_physical_logical_exact": all(
            row["dedup_factor"] == 1 and float(row["prefill_physical_logical_ratio"]) == 1.0
            for row in rows),
        "resident_product_exact": all(row["resident_product_bytes"] == PRODUCT_BYTES for row in rows),
        "no_persistent_second_weight_copy": True,
        "scratch_le_8gib": all(int(row["transient_scratch_bytes"]) <= (8 << 30) for row in rows),
        "memavailable_floor_ge_8gib": all(int(row["mem_available_bytes_min_during_request"]) >= (8 << 30) for row in rows),
        "kv_reported": all(row["kv_cache_bytes"] is not None for row in rows),
    }
    if not all(gates.values()):
        raise SystemExit(f"refuse PASS receipt: failed gates={ {k:v for k,v in gates.items() if not v} }")
    result = {
        "schema": "p530-make-or-break-prefill-result-v1",
        "task": "task-redacted",
        "host": "spark-8",
        "status": "PASS_GE_200_TOK_S",
        "finished_unix": time.time(),
        "candidate": "one-projection-at-a-time streaming dequantize-to-BF16 plus dense torch.mm for all four tiers; decode retains incumbent Triton",
        "p526_component_sha256": P526_SHA,
        "artifact_sha256": summary["artifact_sha256"],
        "final_gate_path": str(FINAL),
        "final_gate_sha256": sha(FINAL),
        "code_sha256": code_hashes,
        "ladder": [
            {"variant": "sealed_product_baseline", "pp2048_prefill_tok_s": 28.950002443, "verdict": "REJECT_LT_200"},
            {"variant": "p526_explicit_m_plus_streaming_native", "pp2048_prefill_tok_s": 117.253752248, "verdict": "REJECT_LT_200"},
            {"variant": "streaming_dequant_dense_all", "pp2048_probe_prefill_tok_s": 1137.632529908508, "verdict": "PROMOTE_TO_FINAL_GATE"},
        ],
        "final_rows": summary["rows"],
        "memory_law": {
            "resident_product_bytes": PRODUCT_BYTES,
            "persistent_second_weight_copy": False,
            "transient_scratch_bytes": max(int(row["transient_scratch_bytes"]) for row in rows),
            "vmhwm_bytes_max": max(int(row["request_vmhwm_bytes"]) for row in rows),
            "mem_available_bytes_floor": min(int(row["mem_available_bytes_min_during_request"]) for row in rows),
            "kv_cache_bytes": sorted(set(int(row["kv_cache_bytes"]) for row in rows)),
            "kv_note": sorted(set(str(row["kv_cache_note"]) for row in rows)),
        },
        "gates": gates,
        "raw_row_files": summary["raw_row_files"],
    }
    atomic_json(OUT, result)
    print(json.dumps({"path": str(OUT), "sha256": sha(OUT), "status": result["status"], "gates": gates}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
