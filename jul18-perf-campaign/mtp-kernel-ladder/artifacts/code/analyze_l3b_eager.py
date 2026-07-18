#!/usr/bin/env python3
"""<source-task>: verify eager vq_warp diagnosis in l3b_cycle.sqlite (node-granular trace).
Steady-state window anchored on the big decode graph replays; report per-cycle cost
of (a) in-graph vq_warp, (b) EAGER vq_warp (graphId NULL/0), (c) deep_gemm class,
(d) grid shapes of eager vq_warp instances."""
import sqlite3, sys
from collections import defaultdict

db = sys.argv[1] if len(sys.argv) > 1 else "$HOME/missions/MTP_L3_<source-task>/profiles/l3b_cycle.sqlite"
con = sqlite3.connect(db)
cur = con.cursor()

def cols(table):
    return [r[1] for r in cur.execute(f"PRAGMA table_info({table})")]

k_cols = cols("CUPTI_ACTIVITY_KIND_KERNEL")
print("kernel cols:", k_cols, file=sys.stderr)

# in node-trace, in-graph kernels carry graphNodeId > 0; eager kernels have 0/NULL
ks = cur.execute("""SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL
  WHERE graphNodeId IS NOT NULL AND graphNodeId > 0 ORDER BY start""").fetchall()
print(f"in-graph kernel instances: {len(ks)}")
# cluster: gap > 5ms starts new replay
replays = []
cs, ce = ks[0]
for s, e in ks[1:]:
    if s - ce > 5e6:
        replays.append((cs, ce)); cs = s
    ce = max(ce, e)
replays.append((cs, ce))
print(f"graph-replay clusters: {len(replays)}")
if len(replays) < 8:
    print("too few replays for steady state"); sys.exit(1)
# steady state: drop first 3 and last 1
ss = replays[3:-1]
t0, t1 = ss[0][0], ss[-1][1]
ncyc = len(ss)
win_ms = (t1 - t0)/1e6
print(f"steady window: {ncyc} cycles, {win_ms:.1f} ms total, {win_ms/ncyc:.2f} ms/cycle")

# demangled names (separate cursor: must not clobber the iteration cursor)
cur2 = con.cursor()
name_cache = {}
def dname(nid):
    if nid not in name_cache:
        r = cur2.execute("SELECT value FROM StringIds WHERE id=?", (nid,)).fetchone()
        name_cache[nid] = r[0] if r else str(nid)
    return name_cache[nid]

# classify all kernels in window
agg = defaultdict(lambda: [0, 0.0])  # (class, in_graph) -> [count, ms]
eager_vq_shapes = defaultdict(lambda: [0, 0.0])
for start, end, nid, gid, gx, gy, gz in cur.execute(f"""
    SELECT start, end, demangledName, graphNodeId, gridX, gridY, gridZ
    FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start >= {t0} AND start < {t1}""").fetchall():
    n = dname(nid)
    ing = bool(gid and gid > 0)
    if "vq_warp" in n:
        klass = "vq_warp"
        if not ing:
            eager_vq_shapes[(gx, gy, gz)][0] += 1
            eager_vq_shapes[(gx, gy, gz)][1] += (end-start)/1e6
    elif "deep_gemm" in n or "fp8" in n.lower() or "nvfp4" in n.lower():
        klass = "deep_gemm/fp"
    elif "gemv" in n.lower() or "wmma" in n.lower() or "cublas" in n.lower() or "cutlass" in n.lower():
        klass = "gemv/wmma"
    else:
        klass = "other"
    agg[(klass, ing)][0] += 1
    agg[(klass, ing)][1] += (end-start)/1e6

print("\nper-cycle kernel cost (class, in_graph): count/cyc, ms/cyc")
for (klass, ing), (c, ms) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
    print(f"  {klass:14s} in_graph={ing}: {c/ncyc:7.2f}/cyc  {ms/ncyc:8.3f} ms/cyc")

print("\nEAGER vq_warp grid shapes: (grid) count/cyc ms/cyc avg_ms")
for shp, (c, ms) in sorted(eager_vq_shapes.items(), key=lambda kv: -kv[1][1]):
    print(f"  {shp}: {c/ncyc:6.2f}/cyc {ms/ncyc:7.3f} ms/cyc  avg {ms/c:.3f} ms")
