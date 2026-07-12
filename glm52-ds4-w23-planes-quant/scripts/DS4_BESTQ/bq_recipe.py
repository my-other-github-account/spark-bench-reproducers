#!/usr/bin/env python3
"""t_3d6e422d final recipe assembly: RECIPE_DS4_BESTQ.json — THE DS4
recipe (manifests + plane dirs + scales + solver provenance + md5s) +
the full KLD ladder snapshot."""
import glob
import hashlib
import json
import os
import time

BQ = os.path.expanduser("~/missions/DS4_BESTQ")
TEACH = os.path.expanduser("~/missions/DS4_TEACHER")
MM = os.path.expanduser("~/missions/DS4_MMLU")


def md5f(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def ledger():
    rows = {}
    for ln in open(f"{TEACH}/KLD_LEDGER.jsonl"):
        r = json.loads(ln)
        rows[r["variant"]] = {k: r[k] for k in
                              ("kl_vs_fp8", "js", "top1_agree",
                               "n_windows", "n_positions") if k in r}
    return rows


def mmlu(tag):
    p = f"{MM}/out/{tag}_MMLU500_ROW.json"
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    return {k: d[k] for k in ("accuracy", "n_correct") if k in d}


def nll(name):
    p = f"{BQ}/out/{name}"
    return json.load(open(p)) if os.path.exists(p) else None


def dir_manifest(d, sample=3):
    metas = sorted(glob.glob(f"{d}/layer_*.meta.json"))
    return {"n_layers": len(metas),
            "meta_layer_000": json.load(open(metas[0])) if metas else None,
            "md5_meta_files": md5f(metas[0]) if metas else None}


def main():
    pa = json.load(open(f"{BQ}/PILOT_ALPHA.json"))
    pg = json.load(open(f"{BQ}/PILOT_GPTAQ.json"))
    m88 = json.load(open(f"{BQ}/R7_MANIFEST_88G.json"))
    m94 = json.load(open(f"{BQ}/R7_MANIFEST_94G.json"))
    led = ledger()
    out = {
        "task": "t_3d6e422d",
        "title": "DS4-Flash best quant, all techniques stacked (KLD-optimal)",
        "ts": round(time.time(), 3),
        "recipe": {
            "w2_tier": {
                "grid": "W2v2-e43: dp_asym4_round2 (t_bd7728ee shootout, "
                        "0.9198x) rounded e4m3 [-3.5,-1.125,0.625,2.75]",
                "scales": "per-block-32 UE8M0 SSE-refit (offsets -4..2) "
                          "vs ckpt mxfp4 exponent",
                "alpha": pa["decisions"],
                "solver": "LUT-GPTQ g4x skeleton (blocksize 128, percdamp "
                          "0.01, colnorm-desc perm, static scales), "
                          "per-proj val-gate 2%, fused13 joint solve, "
                          "down arm + GPTAQ asymmetric-error solve "
                          "(arXiv 2504.02692) adopted="
                          + str(pg["adopt"]),
                "planes_dir": f"{BQ}/planes_gptq_w2v2",
                "rtn_planes_dir": f"{BQ}/moe_w2_planes_v2e43",
            },
            "w3_tier": {
                "grid": "W3v2-e43 dp_asym8_fit rounded e4m3 "
                        "[-6.5,-3.5,-1.875,-0.875,0.140625,1.5,3.5,6.5]",
                "planes_dir": os.path.expanduser(
                    "~/missions/DS4_R6/planes_w3v2_e43"),
                "provenance": "t_eee6b0cc shootout -> t_26055bf3 GPTQ "
                              "-> t_14f51254 e43 anchor",
            },
            "fp4_tier": "ckpt e2m1 block-32 passthrough",
            "allocator": "damage-per-byte greedy = exact knapsack "
                         "(equal 3,145,728-B steps); damage = CALIB-split "
                         "routing mass x shipped-arm activation val relRMS; "
                         "w2 anchor re-measured (R4v2)",
            "manifests": {
                "88G": {"path": f"{BQ}/R7_MANIFEST_88G.json",
                        "md5": md5f(f"{BQ}/R7_MANIFEST_88G.json"),
                        "counts": m88["counts"], "bpw": m88["bpw"],
                        "predicted_kld": m88["predicted_kld"]},
                "94G": {"path": f"{BQ}/R7_MANIFEST_94G.json",
                        "md5": md5f(f"{BQ}/R7_MANIFEST_94G.json"),
                        "counts": m94["counts"], "bpw": m94["bpw"],
                        "predicted_kld": m94["predicted_kld"]},
            },
        },
        "pilots": {
            "alpha": {"gains": pa["gains"], "decisions": pa["decisions"]},
            "gptaq": {"gain": pg["mass_weighted_gain_vs_std"],
                      "adopt": pg["adopt"]},
        },
        "plane_dirs": {
            "moe_w2_planes_v2e43": dir_manifest(
                f"{BQ}/moe_w2_planes_v2e43"),
            "planes_gptq_w2v2": dir_manifest(f"{BQ}/planes_gptq_w2v2"),
        },
        "solver_md5s": {
            os.path.basename(p): md5f(p)
            for p in glob.glob(f"{BQ}/bq_*.py") +
            [f"{BQ}/chain_bestq.sh"]},
        "ledgers": {
            "SOLVE_LEDGER_w2v2": md5f(f"{BQ}/out/SOLVE_LEDGER_w2v2.jsonl"),
        },
        "kld_ladder": led,
        "mmlu": {t: mmlu(t) for t in ("M_Q2v2", "M_Q2gv2", "M_Q7b94")},
        "nll": {n: nll(f) for n, f in
                (("w2v2", "W2V2_NLL.json"), ("w2v2_gptq", "W2V2G_NLL.json"),
                 ("r7_88g", "R7_88G_NLL.json"),
                 ("r7_94g", "R7_94G_NLL.json"))},
        "gates": {"weight_space": json.load(
            open(f"{BQ}/GATE_W2V2.json"))["gates"]},
    }
    p = f"{BQ}/RECIPE_DS4_BESTQ.json"
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print(f"WROTE {p} md5={md5f(p)}")
    print(json.dumps({"ladder": {k: v["kl_vs_fp8"]
                                 for k, v in led.items()},
                      "mmlu": out["mmlu"]}, indent=1))


if __name__ == "__main__":
    main()
