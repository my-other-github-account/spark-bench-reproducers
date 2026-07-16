#!/usr/bin/env python3
"""Build reproducible aggregate and per-window tables from repair JSONL ledgers."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROBE_WINS = [4, 84, 160, 236, 304, 373, 442, 511]


def load_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def summarize(path: Path) -> dict:
    rows = load_rows(path)
    starts = [row for row in rows if row.get("event") == "start"]
    panels = [row for row in rows if row.get("event") == "probe_mean"]
    probes = [row for row in rows if row.get("event") == "probe"]
    completed = any(row.get("event") == "completed" for row in rows)
    start = starts[-1] if starts else {}
    best = max(panels, key=lambda row: float(row["delta_pct"])) if panels else None
    panel_rows = []
    for panel in panels:
        step = int(panel["step"])
        by_win = {int(row["win"]): row for row in probes if int(row.get("step", -1)) == step}
        panel_rows.append(
            {
                "step": step,
                "mean": panel.get("mean"),
                "baseline_mean": panel.get("baseline_mean"),
                "delta_pct": panel.get("delta_pct"),
                "windows": [
                    {
                        "win": win,
                        "baseline": by_win.get(win, {}).get("baseline"),
                        "kld": by_win.get(win, {}).get("kld"),
                        "delta_pct": by_win.get(win, {}).get("delta_pct"),
                    }
                    for win in PROBE_WINS
                ],
            }
        )
    return {
        "tag": start.get("tag", path.stem),
        "source": str(path.relative_to(ROOT)),
        "lr": start.get("lr"),
        "batch": start.get("batch"),
        "planned_steps": start.get("steps"),
        "trainable_layers": len(start.get("trainable", [])),
        "training_windows": len(start.get("train_wins", [])),
        "completed": completed,
        "last_step": max((int(row.get("step", 0)) for row in rows if row.get("event") == "step"), default=0),
        "best_step": int(best["step"]) if best else None,
        "best_delta_pct": best.get("delta_pct") if best else None,
        "best_probe_mean": best.get("mean") if best else None,
        "panels": panel_rows,
    }


def to_markdown(arms: list[dict]) -> str:
    out = [
        "# Repair probe tables",
        "",
        "Generated from the checked-in append-only JSONL ledgers by `summarize_probes.py`.",
        "Positive delta means lower KLD than the exact step-0 baseline. The JSONL rows are authoritative; provisional status-message roundings are not used.",
        "",
        "## Aggregate",
        "",
        "| arm | lr | layers | train windows | progress | best step | best delta | status |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for arm in arms:
        progress = f"{arm['last_step']}/{arm['planned_steps']}"
        best_step = "—" if arm["best_step"] is None else str(arm["best_step"])
        best_delta = "—" if arm["best_delta_pct"] is None else f"{arm['best_delta_pct']:+.4f}%"
        status = "completed" if arm["completed"] else "partial"
        out.append(
            f"| `{arm['tag']}` | {arm['lr']} | {arm['trainable_layers']} | {arm['training_windows']} | "
            f"{progress} | {best_step} | {best_delta} | {status} |"
        )
    for arm in arms:
        out += ["", f"## `{arm['tag']}`", ""]
        if not arm["panels"]:
            out.append("No binding probe panel was sealed; the checked-in ledger is pre-panel/partial.")
            continue
        header = "| step | pooled delta | " + " | ".join(f"w{win}" for win in PROBE_WINS) + " |"
        align = "|---:|---:|" + "---:|" * len(PROBE_WINS)
        out += [header, align]
        for panel in arm["panels"]:
            cells = []
            for row in panel["windows"]:
                value = row["delta_pct"]
                cells.append("—" if value is None else f"{value:+.4f}%")
            out.append(
                f"| {panel['step']} | {panel['delta_pct']:+.4f}% | " + " | ".join(cells) + " |"
            )
    out.append("")
    return "\n".join(out)


def main() -> None:
    paths = sorted(
        path
        for path in ROOT.rglob("BINREPAIR_*.jsonl")
        if "results" in path.parts
    )
    arms = [summarize(path) for path in paths]
    (ROOT / "PROBE_TABLES.json").write_text(json.dumps({"probe_windows": PROBE_WINS, "arms": arms}, indent=2) + "\n")
    (ROOT / "PROBE_TABLES.md").write_text(to_markdown(arms))
    print(f"PROBE_TABLES_WRITTEN arms={len(arms)}")


if __name__ == "__main__":
    main()
