#!/usr/bin/env python3
import json
import math
import re
import statistics as st
import sys
from pathlib import Path

PP_LIST = [2048, 16384, 32768, 65536, 131072]
SIDES = {"dflash": "dflash", "ar-reference": "ar_reference"}
OLD_PP_TARGETS = {
    2048: 2371.62,
    16384: 2781.08,
    32768: 2521.29,
    65536: 2086.81,
    131072: 1543.34,
}


def metrics(values):
    vals = [float(v) for v in values]
    mean = st.mean(vals)
    sd = st.stdev(vals) if len(vals) > 1 else 0.0
    return {
        "mean": mean,
        "median": st.median(vals),
        "min": min(vals),
        "max": max(vals),
        "stdev": sd,
        "cv": sd / mean if mean else math.nan,
        "n": len(vals),
        "values": vals,
    }


def load_benchmark(path):
    data = json.loads(path.read_text())
    return data, data["benchmarks"][0]


def parse_acceptance(log_text):
    matches = list(re.finditer(
        r"SpecDecoding metrics: Mean acceptance length: ([0-9.]+).*?"
        r"Accepted: ([0-9]+), Drafted: ([0-9]+).*?"
        r"Avg Draft acceptance rate: ([0-9.]+)%",
        log_text,
    ))
    if not matches:
        return None
    m = matches[-1]
    return {
        "mean_acceptance_length": float(m.group(1)),
        "accepted": int(m.group(2)),
        "drafted": int(m.group(3)),
        "avg_draft_acceptance_rate_pct": float(m.group(4)),
    }


def side_row(root, side, pp):
    pp_dir = root / side / f"pp{pp}"
    measured = pp_dir / f"measured-pp{pp}-tg128-c1-n30.json"
    warmup = pp_dir / "warmup-pp{}-tg128-c1-n1.json".format(pp)
    data, b = load_benchmark(measured)
    warmup_data, warmup_b = load_benchmark(warmup)
    proof = pp_dir / "server-proof-excerpt.log"
    log_text = proof.read_text(errors="replace") if proof.exists() else ""
    return {
        "artifact": str(measured.relative_to(root)),
        "warmup_artifact": str(warmup.relative_to(root)),
        "server_proof_excerpt": str(proof.relative_to(root)) if proof.exists() else None,
        "shape": {
            "prompt_size": b["prompt_size"],
            "response_size": b["response_size"],
            "concurrency": b["concurrency"],
            "context_size": b["context_size"],
            "is_context_prefill_phase": b["is_context_prefill_phase"],
        },
        "warmup_shape": {
            "prompt_size": warmup_b["prompt_size"],
            "response_size": warmup_b["response_size"],
            "concurrency": warmup_b["concurrency"],
            "context_size": warmup_b["context_size"],
            "is_context_prefill_phase": warmup_b["is_context_prefill_phase"],
        },
        "pp_throughput": metrics(b["pp_throughput"]["values"]),
        "tg_throughput": metrics(b["tg_throughput"]["values"]),
        "acceptance": parse_acceptance(log_text),
        "llama_benchy_version": data.get("version"),
        "prefix_caching_enabled": data.get("prefix_caching_enabled"),
    }


def validate_row(side_name, row, pp):
    errors = []
    shape = row["shape"]
    expected = {
        "prompt_size": pp,
        "response_size": 128,
        "concurrency": 1,
        "context_size": 0,
        "is_context_prefill_phase": False,
    }
    for key, value in expected.items():
        if shape.get(key) != value:
            errors.append(f"{side_name} pp{pp}: {key}={shape.get(key)!r}, expected {value!r}")
    if row["tg_throughput"]["n"] != 30:
        errors.append(f"{side_name} pp{pp}: tg N={row['tg_throughput']['n']}, expected 30")
    if row["pp_throughput"]["n"] != 30:
        errors.append(f"{side_name} pp{pp}: pp N={row['pp_throughput']['n']}, expected 30")
    return errors


def main():
    if len(sys.argv) != 2:
        print("usage: summarize_aeon_paired_allpp.py RESULT_ROOT", file=sys.stderr)
        return 2
    root = Path(sys.argv[1]).resolve()
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    rows = []
    errors = []
    for pp in PP_LIST:
        dflash = side_row(root, "dflash", pp)
        ar = side_row(root, "ar-reference", pp)
        errors.extend(validate_row("dflash", dflash, pp))
        errors.extend(validate_row("ar-reference", ar, pp))
        d_tg = dflash["tg_throughput"]["mean"]
        ar_tg = ar["tg_throughput"]["mean"]
        rows.append({
            "prompt_size": pp,
            "response_size": 128,
            "concurrency": 1,
            "context_size": 0,
            "is_context_prefill_phase": False,
            "dflash": dflash,
            "ar_reference": ar,
            "old_published_dflash_pp_mean_target": OLD_PP_TARGETS[pp],
            "followup_tg_floor_90pct_of_corrected_dflash": 0.9 * d_tg,
            "tg_ratio_dflash_over_ar": d_tg / ar_tg if ar_tg else math.nan,
            "tg_pct_of_ar": 100.0 * d_tg / ar_tg if ar_tg else math.nan,
            "pass_vs_paired_ar": d_tg > ar_tg,
        })

    summary = {
        "metadata": metadata,
        "followup_optimization_targets": {
            "description": (
                "Restore DFlash PP mean to at least the old published PP target "
                "while keeping TG mean >= 90% of this corrected DFlash TG mean."
            ),
            "old_published_dflash_pp_mean_targets": OLD_PP_TARGETS,
            "tg_floor_source": "corrected uploaded paired DFlash TG means from this summary",
        },
        "rows": rows,
        "validation": {
            "passed": not errors,
            "errors": errors,
        },
    }
    out = root / "summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({
        "summary": str(out),
        "passed": not errors,
        "rows": [
            {
                "pp": r["prompt_size"],
                "dflash_tg_mean": r["dflash"]["tg_throughput"]["mean"],
                "ar_tg_mean": r["ar_reference"]["tg_throughput"]["mean"],
                "ratio": r["tg_ratio_dflash_over_ar"],
                "pass": r["pass_vs_paired_ar"],
            }
            for r in rows
        ],
        "errors": errors,
    }, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
