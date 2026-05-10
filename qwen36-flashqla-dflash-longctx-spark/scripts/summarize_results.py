#!/usr/bin/env python3
import json, statistics as st, sys, pathlib
for arg in sys.argv[1:]:
    p = pathlib.Path(arg)
    d = json.loads(p.read_text())
    b = d["benchmarks"][0]
    vals = b["pp_throughput"]["values"]
    print(f"{p}: n={len(vals)} mean={st.mean(vals):.6f} median={st.median(vals):.6f} std={st.pstdev(vals):.6f} min={min(vals):.6f} max={max(vals):.6f}")
    print("values=", vals)
