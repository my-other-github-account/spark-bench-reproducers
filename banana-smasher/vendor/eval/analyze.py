#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import statistics
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PASS = "pass"


def quantile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos)); hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def estimator(n: int, c: int, k: int) -> float:
    if n < k:
        raise ValueError("n < k")
    if n - c < k:
        return 1.0
    product = 1.0
    for value in range(n - c + 1, n + 1):
        product *= 1.0 - float(k) / float(value)
    return 1.0 - product


def load_eval(path: pathlib.Path) -> Dict[str, List[Dict[str, Any]]]:
    value = json.loads(path.read_text())
    rows = value.get("eval")
    if not isinstance(rows, dict) or not rows:
        raise RuntimeError("invalid EvalPlus results %s" % path)
    return rows


def task_metrics(rows: Dict[str, List[Dict[str, Any]]], channel: str) -> Dict[str, Dict[str, Any]]:
    output = {}
    for task_id, samples in rows.items():
        flags = []
        for sample in samples:
            base = sample.get("base_status") == PASS
            plus = sample.get("plus_status") == PASS
            flags.append(base if channel == "base" else (base and plus))
        output[task_id] = {"flags": flags, "n": len(flags), "c": sum(flags), "rate": sum(flags) / len(flags)}
    return output


def aggregate(metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    ns = {row["n"] for row in metrics.values()}
    if len(ns) != 1:
        raise RuntimeError("unequal sample counts")
    n = next(iter(ns))
    values = [row["rate"] for row in metrics.values()]
    result = {"tasks": len(metrics), "n_per_task": n, "pass_at_1": statistics.mean(values)}
    for k in (1, 5, 10, 20):
        if n >= k:
            result["pass_at_%d" % k] = statistics.mean(estimator(row["n"], row["c"], k) for row in metrics.values())
    return result


def bootstrap_mean(values: Sequence[float], rng: random.Random, draws: int = 10000) -> Dict[str, float]:
    n = len(values)
    samples = []
    for _ in range(draws):
        samples.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    return {"estimate": statistics.mean(values), "ci95_low": quantile(samples, 0.025), "ci95_high": quantile(samples, 0.975), "draws": draws}


def paired_permutation(deltas: Sequence[float], rng: random.Random, draws: int = 100000) -> Dict[str, Any]:
    observed = abs(statistics.mean(deltas))
    nonzero = [value for value in deltas if value != 0]
    if not nonzero:
        return {"method": "degenerate_all_ties", "p_two_sided": 1.0, "nonzero_tasks": 0}
    if len(nonzero) <= 20:
        extreme = 0; total = 1 << len(nonzero)
        for mask in range(total):
            value = sum((-item if (mask >> idx) & 1 else item) for idx, item in enumerate(nonzero)) / len(deltas)
            extreme += abs(value) >= observed - 1e-15
        return {"method": "exact_sign_flip", "p_two_sided": extreme / total, "permutations": total, "nonzero_tasks": len(nonzero)}
    extreme = 0
    for _ in range(draws):
        value = sum(item if rng.random() < 0.5 else -item for item in nonzero) / len(deltas)
        extreme += abs(value) >= observed - 1e-15
    return {"method": "monte_carlo_task_sign_flip", "p_two_sided": (extreme + 1) / (draws + 1), "permutations": draws, "nonzero_tasks": len(nonzero)}


def sign_test(deltas: Sequence[float]) -> Dict[str, Any]:
    positive = sum(value > 0 for value in deltas); negative = sum(value < 0 for value in deltas); n = positive + negative
    if n == 0:
        return {"positive": 0, "negative": 0, "ties": len(deltas), "p_two_sided": 1.0}
    tail = sum(math.comb(n, k) for k in range(0, min(positive, negative) + 1)) / (2 ** n)
    return {"positive": positive, "negative": negative, "ties": len(deltas) - n, "p_two_sided": min(1.0, 2 * tail)}


def classify_timing(matrix: Dict[str, pathlib.Path]) -> Dict[str, Any]:
    loaded = {name: load_eval(path) for name, path in matrix.items()}
    ids = set(loaded["canonical"])
    rows = []
    counts: Dict[str, int] = {}
    for task_id in sorted(ids):
        n = len(loaded["canonical"][task_id])
        for sample_index in range(n):
            statuses = {cell: {"base": loaded[cell][task_id][sample_index].get("base_status"), "plus": loaded[cell][task_id][sample_index].get("plus_status")} for cell in loaded}
            canonical = statuses["canonical"]
            relaxed = statuses["relaxed"]
            if canonical["base"] == canonical["plus"] == PASS:
                label = "pass"
            elif canonical != relaxed and relaxed["base"] == relaxed["plus"] == PASS:
                label = "timing_only"
            elif any("timeout" in str(value).lower() or "time" in str(value).lower() for pair in statuses.values() for value in pair.values() if value is not None):
                label = "timing_marginal_unresolved"
            else:
                label = "semantic_or_runtime_failure"
            counts[label] = counts.get(label, 0) + 1
            if label != "pass":
                rows.append({"task_id": task_id, "sample_index": sample_index, "classification": label, "statuses": statuses})
    return {"counts": counts, "nonpass_rows": rows}


def uncertainty_rows(rows_root: pathlib.Path, model: str, dataset: str, decode_arm: str, eval_rows: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    records = []
    from p968_common import row_path
    for task_id, samples in eval_rows.items():
        for sample_index, result in enumerate(samples):
            path = row_path(rows_root, model, dataset, decode_arm, sample_index, task_id)
            value = json.loads(path.read_text())
            gaps = [gap for gap in value.get("uncertainty", {}).get("top1_top2_gaps", []) if gap is not None]
            median_gap = statistics.median(gaps) if gaps else None
            passed = result.get("base_status") == result.get("plus_status") == PASS
            records.append({"task_id": task_id, "sample_index": sample_index, "median_top_gap": median_gap, "pass": passed, "available": median_gap is not None})
    available = [row for row in records if row["available"]]
    threshold = quantile([row["median_top_gap"] for row in available], 0.5) if available else None
    confident = [row for row in available if row["median_top_gap"] >= threshold] if threshold is not None else []
    return {"definition": "within-model upper half by per-completion median top1-top2 token logprob gap", "available": len(available), "threshold": threshold, "confident": len(confident), "confident_pass_rate": (sum(row["pass"] for row in confident) / len(confident) if confident else None), "confident_false_positive_rate": (sum(not row["pass"] for row in confident) / len(confident) if confident else None)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--seed", type=int, default=968)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    models = config["models"]
    rng = random.Random(args.seed)
    report: Dict[str, Any] = {"schema": "p968-statistical-audit-v1", "seed": args.seed, "models": {}, "paired": {}, "timing": {}}
    normalized: Dict[Tuple[str, str, str], Dict[str, List[Dict[str, Any]]]] = {}
    for model, datasets in models.items():
        report["models"][model] = {}
        for dataset, cell in datasets.items():
            sampled = load_eval(pathlib.Path(cell["sampled_canonical"]))
            greedy = load_eval(pathlib.Path(cell["greedy_canonical"]))
            normalized[(model, dataset, "sampled")] = sampled
            normalized[(model, dataset, "greedy")] = greedy
            channels = {}
            for channel in ("base", "plus"):
                sm = task_metrics(sampled, channel); gm = task_metrics(greedy, channel)
                sampled_agg = aggregate(sm); greedy_agg = aggregate(gm)
                sampled_agg["cluster_bootstrap_pass_at_1"] = bootstrap_mean([row["rate"] for row in sm.values()], rng)
                unstable = [task_id for task_id, row in gm.items() if 0 < row["c"] < row["n"]]
                greedy_agg["unstable_tasks"] = unstable
                greedy_agg["unstable_task_count"] = len(unstable)
                channels[channel] = {"sampled": sampled_agg, "greedy": greedy_agg}
            channels["conditional_confidence"] = uncertainty_rows(pathlib.Path(cell["rows_root"]), model, dataset, "sampled", sampled)
            report["models"][model][dataset] = channels
            report["timing"][model + ":" + dataset + ":sampled"] = classify_timing({name: pathlib.Path(path) for name, path in cell["sampled_matrix"].items()})
            report["timing"][model + ":" + dataset + ":greedy"] = classify_timing({name: pathlib.Path(path) for name, path in cell["greedy_matrix"].items()})
    left, right = config["comparison"]
    for dataset in sorted(set(models[left]) & set(models[right])):
        report["paired"][dataset] = {}
        for arm in ("sampled", "greedy"):
            report["paired"][dataset][arm] = {}
            for channel in ("base", "plus"):
                lm = task_metrics(normalized[(left, dataset, arm)], channel); rm = task_metrics(normalized[(right, dataset, arm)], channel)
                common = sorted(set(lm) & set(rm))
                deltas = [lm[task_id]["rate"] - rm[task_id]["rate"] for task_id in common]
                report["paired"][dataset][arm][channel] = {"direction": left + " minus " + right, "task_delta": bootstrap_mean(deltas, rng), "sign_test": sign_test(deltas), "paired_permutation": paired_permutation(deltas, rng), "tasks": len(common)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "out": str(args.out)}, sort_keys=True))


if __name__ == "__main__":
    main()
