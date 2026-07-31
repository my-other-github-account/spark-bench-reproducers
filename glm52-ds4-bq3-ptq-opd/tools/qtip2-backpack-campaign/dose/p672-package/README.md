# P672 p13 activation-cache pipeline overlay

This stopped-mission-only bundle applies the measured P672 p13 payload/decode
pipeline on top of the exact P649 `r4_c1_p2_m64n32w8` postimage.

Required runtime environment:

```text
P672_P13_PIPELINE=1
P672_P13_GROUP=1
P649_EXPERT_RESIDENT_SCOPE=4
P649_DEQ_CHUNK=1
P649_NATIVE_CHUNK=1
P649_P2_BLOCK_M=64
P649_P2_BLOCK_N=32
P649_P2_NUM_WARPS=8
BANANA_SMASHER_REPAIR_MEM_FLOOR_BYTES=34359738368
```

Usage (target must be stopped):

```bash
python adopt_p13_pipeline.py status /path/to/mission
python adopt_p13_pipeline.py apply /path/to/mission
python adopt_p13_pipeline.py rollback /path/to/mission
python verify_bundle.py
```

Measured canonical receipt:
- target p13 payload/decode: 108.552647s -> 56.400690s (1.924669x)
- routed total: 351.039807s -> 298.887850s
- full43 wall: 360.275995s
- full43 cadence: 8.378512s/layer
- MemAvailable floor: 36.824814 GiB
- minutes saved per 24 updates vs P649: 2.371583

The legacy P649 payload path remains in the source as the exact oracle and can
also be selected by setting `P672_P13_PIPELINE=0`. Atomic rollback restores the
exact P649 physical source SHA.
