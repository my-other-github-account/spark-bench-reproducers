#!/usr/bin/env python3
"""R6 mixed-tier prepack, s8 side, arm-parametrized (t_cf38c8c9 SPEC UPDATE).
Same verbatim row-slice extraction as r6_extract_256k.py (sealed lineage
t_14f51254), but takes manifest/md5/outdir as CLI args so the 94G and 96G
arms share one script.

usage: r6_extract_arm.py <manifest.json> <expected_md5> <out_dir>
"""
import hashlib
import json
import os
import sys

import numpy as np

GPTQ = os.path.expanduser("~/missions/DS4_GPTQ")
TIER_DIR = {"w2": os.path.join(GPTQ, "planes_gptq_w2"),
            "w3": os.path.join(GPTQ, "planes_gptq_w3v2")}
TAGS = ("planes13", "sc13", "planes2", "sc2")
TIER_CODE = {"w2": 0, "w3": 1, "fp4": 2}


def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def main():
    manifest, md5_want, out = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out, exist_ok=True)
    mm = md5_file(manifest)
    assert mm == md5_want, f"manifest md5 drift: {mm} want {md5_want}"
    man = json.load(open(manifest))
    assign = man["assignment"]

    md5s = {}
    for Ls in sorted(assign, key=int):
        L = int(Ls)
        dst = os.path.join(out, f"layer_{L:03d}")
        if os.path.exists(dst + ".meta.json"):
            print(f"layer {L}: exists, skip", flush=True)
            continue
        amap = assign[Ls]
        E = len(amap)
        tier_of = [TIER_CODE[amap[str(e)]] for e in range(E)]
        idx = {t: [e for e in range(E) if amap[str(e)] == t]
               for t in ("w2", "w3", "fp4")}
        slot_of = [-1] * E
        for t, lst in idx.items():
            for i, e in enumerate(lst):
                slot_of[e] = i
        assert min(slot_of) >= 0

        meta_src = None
        for t in ("w2", "w3"):
            rows = idx[t]
            src_pre = os.path.join(TIER_DIR[t], f"layer_{L:03d}")
            meta_src = json.load(open(src_pre + ".meta.json"))
            if not rows:
                continue
            ridx = np.asarray(rows, dtype=np.int64)
            for tag in TAGS:
                src = np.load(f"{src_pre}.{tag}.npy", mmap_mode="r")
                sub = np.ascontiguousarray(src[ridx])
                fn = f"{dst}.{t}.{tag}.npy"
                np.save(fn, sub)
                back = np.load(fn, mmap_mode="r")
                for k in (0, len(rows) - 1):
                    assert np.array_equal(back[k], src[rows[k]]), \
                        f"L{L} {t} {tag} row {k} read-back mismatch"
                md5s[os.path.basename(fn)] = md5_file(fn)
                del src, sub, back

        meta = dict(
            mixed=True,
            E=E, N13=meta_src["N13"], K13=meta_src["K13"],
            N2=meta_src["N2"], K2=meta_src["K2"],
            counts={t: len(idx[t]) for t in ("w2", "w3", "fp4")},
            tier_of=tier_of, slot_of=slot_of,
            manifest_md5=md5_want,
            w2_src="planes_gptq_w2 (t_fa509f27)",
            w3_src="planes_gptq_w3v2 (t_26055bf3, dp_asym8 LUT; serve cubins "
                   "= moe_w3_mm_e43 nearest-e4m3 pool R27=0xb6bfc6cd "
                   "R13=0x4d463c21)",
            fp4_src="s7-local ckpt e2m1+UE8M0 verbatim (r6_fp4_pack_arm.py)",
            task="t_cf38c8c9",
            variant=man.get("variant"),
        )
        with open(dst + ".meta.json.tmp", "w") as f:
            json.dump(meta, f)
            f.flush()
            os.fsync(f.fileno())
        os.rename(dst + ".meta.json.tmp", dst + ".meta.json")
        md5s[f"layer_{L:03d}.meta.json"] = md5_file(dst + ".meta.json")
        print(f"layer {L}: w2={len(idx['w2'])} w3={len(idx['w3'])} "
              f"fp4={len(idx['fp4'])} extracted", flush=True)

    with open(os.path.join(out, "S8_EXTRACT.md5"), "a") as f:
        for k in sorted(md5s):
            f.write(f"{md5s[k]}  {k}\n")
    print("R6MIX_EXTRACT_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
