# P963 exact acceleration rail

P963 accelerated the sealed P951 exact TRUE-C BALANCED64 read from 3,643.123104 seconds to 1,495.797174 seconds: 2.435573x speedup and 58.941899% wall-clock reduction. The accelerated and baseline output-set SHA-256 are identical (`3529d33893a12d92dda96beba29c1a0e21adec6d008f2b32ced7d0066662c451`), the maximum absolute per-position delta is 0.0, and all 64/64 tensors match.

The public artifacts preserve every decision-bearing numerical field while replacing private mission paths, task identifiers, node names, and addresses with explicit placeholders. `artifacts/ARTIFACT_PROVENANCE.json` binds each source file SHA to its public-copy SHA.

## Published files

- `artifacts/P963_EXACT_ACCELERATION_SEAL.public.json`: terminal exactness and speed seal.
- `artifacts/P963_ACCEL_EXACT_P951_BALANCED64_V3_MB2.public.json`: full accelerated measurement receipt.
- `artifacts/P963_BATCHED_STAGE_L000_CANARY.public.json`: 8.400880x transfer-stage canary.
- `code/p963_true_c_accel.py`: measured rail runner; source SHA `44ff2771fad236ad9d25fdbcd4ccdbfdb24b0725a27631650eb9748cb50cfdf8`.
- `code/p963_true_c_overlay_adapter.py`: batched/double-buffered exact-overlay adapter; source SHA `e84efd6080806ca51bf8681e05e7e06aef6d2406bab29da4d4b68ff8d551415e`.
- `code/launch_p963.sh`: launcher; source SHA `393070f9b8c6184f062a9c5cf42f4712492174a7619f4768c3063305ed412c30`.

## Adoption path

1. Bind `/PUBLIC_SOURCE_ROOT` and `PUBLIC_NODE_*_ADDRESS` placeholders to an operator-controlled replica of the exact SHA-pinned inputs.
2. Keep microbatch 2. Microbatch 4 was faster but failed the `<=1e-12` decision gate and is not admissible.
3. Retain the lever order: resident-once objects, mmap for partial layers, one batched peer stage per node, overlap/double buffering, exact-full-overlay base-fill skip, then the highest numerically safe microbatch.
4. Resolve all runtime inputs through the immutable-SHA guard before staging; do not treat historical paths or basenames as authority.
5. Run the L000 batched-stage canary and require exact expected-SHA namespace verification before a full read.
6. Re-run BALANCED64 against the sealed P951 tensor set. Require identical output-set SHA, max absolute tensor delta 0.0 (hard ceiling `1e-12`), 64/64 coverage, pack fraction 1.0, and zero substitution/quarantine.
7. Only after those gates pass may the acceleration rail become the default read path. The published scripts are an adoption reference, not an automatic fleet mutation.
