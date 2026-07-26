#!/usr/bin/env python3
import argparse
import json
import os
import signal
import time
from pathlib import Path


def atomic_json(path: Path, obj):
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def memavailable():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable missing")


def pgid_rss(pgid):
    page = os.sysconf("SC_PAGE_SIZE")
    total = 0
    pids = []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            fields = (d / "stat").read_text().split()
            if int(fields[4]) != pgid:
                continue
            rss = int(fields[23]) * page
            total += rss
            pids.append({"pid": int(d.name), "rss_bytes": rss})
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError, IndexError):
            pass
    return total, pids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watched-pid", type=int, required=True)
    ap.add_argument("--pgid", type=int, required=True)
    ap.add_argument("--floor-bytes", type=int, required=True)
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--stop-json", required=True)
    a = ap.parse_args()
    tsv = Path(a.tsv)
    tsv.parent.mkdir(parents=True, exist_ok=True)
    with tsv.open("a", buffering=1) as out:
        if out.tell() == 0:
            out.write("epoch_ns\tmemavailable_bytes\tpgid_rss_bytes\tpids_json\n")
        while True:
            try:
                os.kill(a.watched_pid, 0)
            except ProcessLookupError:
                return
            epoch_ns = time.time_ns()
            avail = memavailable()
            rss, pids = pgid_rss(a.pgid)
            out.write(f"{epoch_ns}\t{avail}\t{rss}\t{json.dumps(pids, separators=(',', ':'))}\n")
            if avail < a.floor_bytes:
                atomic_json(Path(a.stop_json), {
                    "schema": "two-host-memory-floor-stop-v1",
                    "epoch_ns": epoch_ns,
                    "memavailable_bytes": avail,
                    "floor_bytes": a.floor_bytes,
                    "pgid": a.pgid,
                    "pids": pids,
                    "status": "STOP_MEMORY_FLOOR",
                })
                try:
                    os.killpg(a.pgid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                return
            time.sleep(1)


if __name__ == "__main__":
    main()
