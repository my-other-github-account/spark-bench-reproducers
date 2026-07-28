#!/usr/bin/env python3
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
pid = int(sys.argv[1])
path = ROOT / "out/PROGRESS.json"
try:
    doc = json.loads(path.read_text()) if path.exists() else {}
except Exception:
    doc = {}
proc = Path(f"/proc/{pid}")
live = proc.exists()
metrics = {"pid": pid, "live": live}
if live:
    stat = (proc / "stat").read_text().split()
    ticks = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
    metrics.update(
        {
            "ppid": int(stat[3]),
            "pgid": int(stat[4]),
            "sid": int(stat[5]),
            "cpu_seconds": (int(stat[13]) + int(stat[14])) / ticks,
            "rss_bytes": int(stat[23]) * os.sysconf("SC_PAGE_SIZE"),
            "elapsed_seconds": time.monotonic() - (int(stat[21]) / ticks),
        }
    )
doc.update(
    {
        "schema": "p637-progress-v1",
        "solver_process": metrics,
        "updated_unix": time.time(),
    }
)
tmp = path.with_name(path.name + f".tmp.monitor.{os.getpid()}")
tmp.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
os.replace(tmp, path)
