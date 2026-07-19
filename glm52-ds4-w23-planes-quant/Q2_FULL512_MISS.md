# Q2 fresh-200 squeeze: full-512 miss

Updated 2026-07-19. This is a sealed negative at the 95.75 GB target, not an extrapolation.

## Training result

The arm warm-restarted from the exact step-45 Q2 checkpoint, reset cosine at half the original peak, applied 3x codebook LR, and used 200 sealed disjoint fresh teacher windows. The probe trend was:

| step | held-out probe mean KLD |
|---:|---:|
| 0 | 0.0545097251 |
| 5 | 0.0556653562 |
| 10 | 0.0553876181 |
| 15 | 0.0553879112 |
| 20 | 0.0554114261 |
| 25 | 0.0550787516 |
| 30 | 0.0543547014 |
| 35 | 0.0542473942 |
| **40** | **0.0539240856** |
| 45 | 0.0542526336 |

Step 40 was protected as the best probe checkpoint, improving 1.074376% over step0/source. Checkpoint SHA-256:

```text
62e250db2cdd3c9a13ae50f71637afd6554c743c5dc18eb384e1826ba9251d8d
```

The best probe did **not** transfer to the full corpus.

## Exact full-512 measurement

The accelerated rail first passed an 8-window scalar-versus-accelerated gate: 8/8 payload MD5 matches and 8/8 exact NLL matches. The sealed measurement then covered 512 windows / 524,288 scored positions.

| metric | measured |
|---|---:|
| mean KL versus source teacher | **0.09848245703125** |
| JS | 0.021281802734375 |
| top-1 agreement | 0.9050960390625 |
| top-1-in-teacher-top64 | 0.999414291015625 |
| package | 95.75 GB decimal |
| effective bpw | 3.4471773954 |
| strict target | <0.0927 |
| miss | +0.0057824570 / **6.2378%** |

Block-of-64 means, in window order:

```text
0.096752125
0.103551046875
0.10241528125
0.09002865625
0.0872209375
0.090176796875
0.115011828125
0.102702984375
```

The block spread is material; a favorable small probe was not a substitute for the complete rail.

## Provenance

- measured-row SHA-256: `a08e66fd22c29a1e48afdd6ce134aa7f6d57b33a9549c396f84d57d35ff65dcb`
- export receipt: `e3fc26b7461235fb40b1d7197f528597de84bb2b67e5a61dc1d73f8a8110df53`
- accelerated gate8: `d773d3f8e1fb3dfe8421375c64490b5ac1d765f895fb4a20d115b3057155dcf3`
- score ledger: `7544cc7a3de0721296fb857f18acff336ac120f8bf16c1bcf456aea6e2277f6c`
- canonical corpus MD5: `1701920b4ba96dea0b18fe9df0151876`

## Decision and resume state

**Park Q2.** Do not ship or continue squeezing based only on the 0.053924 probe. The complete 512-window row misses the bar.

If reopened, resume from the protected step-40 checkpoint above and the sealed fresh-200 layout. Distinct next levers are:

1. more disjoint data, not replay of the same 200 windows;
2. an explicitly preregistered codebook-LR schedule sweep;
3. a class/trajectory objective borrowed from BIN T only with clean splits;
4. complete full-512 adjudication for every newly selected checkpoint.

Do not rerun the already sealed full-512 checkpoint. The measured row is terminal for this artifact.

## Reproduction template

```bash
MISSION_ROOT="$CAMPAIGN_ROOT/q2/repro" \
ARM4_CKPT="$MODEL_ROOT/q2-step40" \
REF_LEDGER="$CORPUS_ROOT/ledger_512.json" \
bash "$CAMPAIGN_ROOT/tooling/rail512_shard.sh" "0,1,2,3,4,5,6,7" q2_gate8
```

The command is schematic because the public campaign does not distribute model weights or teacher-row caches. Reproduction must bind the checkpoint SHA, corpus MD5, scorer SHA, 8-window exactness gate, and all 512 unique window IDs before accepting a result.
