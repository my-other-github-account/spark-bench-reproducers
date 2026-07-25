# Accelerated components and adoption map

This is the public, infrastructure-neutral component ledger frozen from the P602 lineage. Private host paths and task identifiers are intentionally omitted; receipt SHA-256 identities are the durable references.

## Accelerated components

| component | measured gain/status | public provenance | consumers |
|---|---:|---|---|
| QTIP batch-v3 quantizer | 35.15x | qtip-canonical commit `e90c6688`; fast-viterbi SHA prefix `79a645f7` | all QTIP unit production |
| builder, tmpfs two-slot | 162.848 s/layer; PASS | source `7f863344ebd999ecdd9a0e132d56bfcbf60b3fd5cc8421cc97d3443516a8837e`; receipt `ab2be95a8e1396f067b0c1cd132152988d37234a8926981aa4b24ca131363bd5` | all physical wire builds |
| torch-native k-means | 11.80x, inertia +0.56% | builder acceleration package | builder and profile fits |
| torch-mmap loader | 3.2x/layer class | builder and rail source | every plane/checkpoint consumer |
| Eval/KLD kernel | 3.886x kernel | 258.5 to 66.5 ms per 8,192 rows, max delta 0.0002 | rail, profile, probes |
| teacher postprocessing | 5.21x | postprocessing only, not generation | teacher-bank postprocessing |
| canonical repair trainer | 520.314 / 520.249 s measured updates | result `eb31eeec746901d77cd921e9b9e7a66e947bd846187bb3ce8b0f39dd3a2e0950` | repair stage |
| SM121 BF16 sparse-MLA fallback | serving unblock | hash-guarded torch-SDPA plain-row fallback | DeepSeek-V4 GB10 serving |
| QTIP packed serve backend | 24.390 tok/s proof | packed QTIP layer plus official MoE patch | serving lanes |
| mixed-tier dense-all prefill | 1,142.085 tok/s at PP2048 | P530 result and ladder under `receipts/p530/` | product-scale systems serving |

## Adoption-gap ledger

| win | kernel gain | realized gain | state |
|---|---:|---:|---|
| Eval/KLD hook in rail system path | 3.886x | 1.60x | system-stage decomposition open |
| builder pilot to tmpfs certification | 3.862x | 2.81-2.89x | closed by tmpfs certification |
| k-means/KLD in profile path | 11.80x / 3.886x | unaudited | adoption audit open |
| k-means/KLD in probe arms | not isolated | unaudited | adoption audit open |

A sealed kernel win is not a pipeline win until the production consumer imports it and the consumer wall moves. Every new acceleration must include a consumer-adoption receipt, not only a microbenchmark.

## Dead levers

Do not retry these without new evidence:

- teacher torch-mmap resident-logits generation: 0.188x, 11.16 h projected;
- the retired vLLM teacher-bank route: host-fatal NVRM out-of-memory;
- resident/EngineCore persistent rail evaluators: host-fatal;
- Eval parallel-4 serve: 0.988x plus context-length HTTP failures;
- repair without checkpointing at microbatch 1 or 2: out of memory near the unified-memory limit;
- grouped routed-expert 0.903x, persistent grouped 0.626x, code-major no-sort 0.777x, single-expert fused 0.990x;
- builder remote-QSFP write: 246.156 s/layer versus 162.848 s/layer with tmpfs;
- builder nlist/nprobe tuning: 217.297 s/layer;
- fastsafetensors on GB10 unified memory: slower than torch-mmap;
- faiss-gpu k-means: +9.078% inertia; cuML: invalid.
