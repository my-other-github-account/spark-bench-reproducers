# Procedures and Operating Laws

Public operator identity: **banana_bae**

These are the campaign procedures that survived actual failure. They are written as fail-closed rules: if a required proof is absent, do not launch the next expensive or destructive stage.

## 1. Public identity and receipt law

Public issues, commits, comments, manifests, and docs use only `banana_bae`. Replace local usernames, machine paths, LAN addresses, private hostnames, board task identifiers, and private mission roots with role labels or variables.

Every scientific claim binds:

- schema/version;
- input SHA-256 values;
- code/config SHA-256 values;
- model/runtime lineage;
- instrument/window SHA-256;
- exact coverage counts;
- output SHA-256 values;
- status: measured, projected, estimated, diagnostic, or pending.

Do not shorten a hash when it is the only integrity key.

## 2. Frozen instrument law

Cross-wire comparison is valid only when all of these match:

- teacher/model lineage;
- tokenizer and support;
- KL direction;
- window IDs and order;
- cutoff and positions/window;
- reducer and class mapping;
- exact candidate checkpoint lineage.

For this package the canonical instrument is BALANCED64_V1: 64 windows, 65,536 positions, support 8,192, cutoff 1,024, `KL(teacher || candidate)`, window-manifest SHA-256 `7f756b898aea80cb4dd9320da4cd0c855f258d055f62ef6c37151d27857fa0ad`.

If one field changes, label the row a different instrument. IQ4 is a useful benchmark reference, but it is not a same-cell-population Wire-C comparison.

## 3. Pin once, consume locally

Before a run, pin each authority exactly once. The scorer/solver/build must consume those task-local pinned copies; it must not rediscover “latest” inputs by scanning mutable trees.

Required pin classes:

1. assignment/map;
2. per-cell tier/menu surface;
3. source payload and codebook manifests;
4. model/tokenizer/runtime;
5. instrument/window manifest;
6. solver/build/scorer code and config;
7. immutable base-wire manifest.

A task-local input manifest is the only runtime menu. External rescans, refilters, silent tier substitutions, and “newer available” data are forbidden after pinning.

## 4. Bulk transfer law

### 4.1 Route and staging

- Tensor payloads move compute-to-compute over the payload fabric.
- Control traffic may use management links.
- Stage all required payloads on compute-local disk before GPU launch.
- Hash at source, destination, and first consumption.
- Do not stream tensors during scoring or reconstruction.
- NAS is archive/backup only; never use it as a live bulk source or working directory.

### 4.2 Measured table

| receipt | shape | observed/gate | use |
|---|---|---:|---|
| direct 30 GB fabric probe | contiguous | >5.0 GB/s | fabric/tool health |
| 8-way directory transfer | large tree | ~2 GB/s | default bulk recipe |
| 95 GB verified base stage | archive+hash+extract | ~0.158 GB/s end-to-end | planning only; includes non-network work |
| 4-way tar+TCP | large archive groups | ≥2 GB/s gate | alternate bulk recipe |

The ≥2 GB/s campaign payload floor and >5 GB/s clean fabric-probe floor are different gates. Do not confuse verified/extract throughput with raw link capability.

### 4.3 Directory-tree recipe

```bash
find "$SRC" -mindepth 1 -maxdepth 1 -print0 \
  | xargs -0 -P8 -I{} rsync -a --partial -e 'ssh -o BatchMode=yes' "{}" "$DEST/"
```

### 4.4 Large-tar recipe

Split the file list over four ports. Receiver pattern:

```bash
nc -l "$PORT" | tar -xf - -C "$DEST"
```

Sender pattern:

```bash
tar -cf - $FILES | nc -N "$RECEIVER" "$PORT"
```

Use four disjoint file lists/ports in parallel. Verify byte counts and SHA-256 after transfer.

### 4.5 Prohibitions

- No single-stream transfer above 10 GB.
- No tensor payloads via a laptop/controller.
- No NAS fallback because a bulk command is slow.
- No GPU launch before local payload closure.

## 5. FORTRESS durability law

The campaign lost useful work because a blocked run’s scratch workspace disappeared and some upper payloads existed only in shm. FORTRESS prevents recurrence.

### 5.1 Durable-payload invariant

A payload counts only when:

1. bytes are on durable compute-local disk;
2. the write is atomic or followed by fsync;
3. size and SHA-256 match the pinned expected value;
4. a progress ledger names the exact identity;
5. a layer seal or checkpoint can reconstruct coverage without process memory.

`/dev/shm` may be a disposable cache. It is never the only authority for an accepted payload.

### 5.2 Checkpoint hierarchy

- Per cell: atomic disk output + receipt.
- Every 64 cells: fsynced exact identity/hash checkpoint.
- Per layer: deterministic manifest, zero-gap/duplicate audit, full reread.
- Per partition: peer copy or durable second readback before retirement.
- Terminal: 43-layer completeness ledger with selected/excluded duplicate provenance.

Keep a rolling two-deep set of sealed layers until peer verification completes. Never delete a sealed producer copy before its consumer or peer receipt closes.

### 5.3 Controller contract

Long GPU work uses one task-owned detached controller with explicit PID/PGID/SID, start time, command, log, claim, and checkpoint paths. The script—not an external timeout—implements retry/resume and resource guards.

Forbidden for ad-hoc campaign work: system services, systemd-run, tmux as an untracked authority, timeout-kills, or shm-only detached workers.

## 6. Single-owner orphan-release law

A blocked or completed card may not own a live workload, host claim, GPU allocation, or unsealed payload.

At every lifecycle transition choose exactly one:

1. **Release:** stop exact owned processes, prove GPU/ports/process scope empty, seal restart state, and exact-release the host claim.
2. **Successor:** name one live successor, transfer the exact claim/process/checkpoint facts, and have the successor adopt before the old owner exits.

“Someone will pick it up” is not a handoff.

## 7. Orphan adoption procedure

Adopt a detached workload only after all gates pass:

1. Read the live claim and verify exact task ownership.
2. Verify PID, PGID, SID, process start time, executable, command, and parentage.
3. Verify log and progress motion.
4. Verify the newest checkpoint parses and all named payload hashes exist.
5. Verify no competing live owner/card holds the same partition.
6. Record an adoption receipt before mutating the process.
7. Resume the existing workload; do not relaunch completed stages.

If the board lifecycle says the previous run is stale while the physical controller is live, the stale shell becomes read-only. A fresh owner performs adoption.

## 8. Speculative parallel-work law

Parallelism is permitted only when the work is physically partitioned and arbitration is defined before launch.

Required fields per lane:

- exact host/partition/layer bounds;
- source/config SHA-256;
- first durable payload deadline;
- checkpoint frequency;
- stop boundary;
- winner rule;
- duplicate handling;
- terminal release owner.

For layer work, use **first complete sealed layer wins**. Never kill a producer mid-layer merely because another lane appears ahead. At the next atomic seal, select one authoritative receipt, mark duplicates as recovery-only, and repartition before either lane enters the next layer.

A signal that is swallowed by a loaded runtime is not a stop. Confirm process absence or use exact PID/start-time escalation at the defined boundary.

## 9. Solver law

### 9.1 Current snapshot is the menu

The solver may select only options present in the frozen task-local menu. Missing tiers remain unavailable. Do not add newer data or rebuild a historical menu from mutable sources.

### 9.2 Gate failure and preview overrides

A failed retrodiction gate blocks a production solve/build. An explicit operator override may authorize a preview only if every output says:

- PREVIEW;
- which gate failed;
- why the result may be biased;
- which definitive rerun is still required.

### 9.3 Dominance pruning

Drop an option only if another option for the same identity is no worse in every constrained price component, no larger in bytes, and strictly better in at least one dimension. Preserve original and warm-hint options.

### 9.4 Greedy-zero and paired-swap rescue

A one-cell greedy search can return zero because exact byte coupling makes every useful upgrade infeasible alone. Do not interpret zero as optimal.

Deterministic paired-swap procedure:

1. Enumerate freeing moves and spending moves by exact identity.
2. Pair a freeing move with an upgrade whose net bytes fit.
3. Evaluate the pair atomically against global and every class cap.
4. Require strict objective improvement.
5. Tie-break by `(gain, net_bytes, layer, expert, projection, old_tier, new_tier)` with a documented stable direction.
6. Apply the best pair, update residuals, and repeat.
7. Replay the full assignment after each accepted batch.
8. Use the result as a solver hint, never as an unverified final receipt.

LP relaxation plus exact repair is also valid: solve the relaxation, round deterministic basic variables, and run a tiny exact MILP over the remaining fractional groups. Publish the lower bound and gap; do not claim optimality without proof.

## 10. Build/reconstruction law

- Build from an immutable sealed base plus only changed cells.
- Verify exact expected changed-cell coverage before GPU launch.
- Preserve pack/runtime metadata at 1.0 unless the pinned spec says otherwise.
- Every changed VQ row must bind an exact local codebook by path/size/SHA.
- Reconstruct the full assignment from base + overlay and compare identity-by-identity.
- Treat assignment version drift as a hard failure; rebuild the affected layer rather than patching receipts.

## 11. Scoring law

Before scoring:

- stage teacher, wire, scorer, tokenizer, and BALANCED64 inputs locally;
- verify their hashes;
- prove exact changed-cell and layer coverage;
- run one frozen-window sanity gate;
- record model/runtime split and resource guard receipts.

After scoring:

- publish global and all six class rows;
- publish window counts, per-window rows, SE/CI, and direction;
- independently reread/reduce the output;
- label the result measured, diagnostic, estimated, or pending;
- exact-release the host.

The P922 restored-VQ diagnostic is not TRUE-C. Only the dependency-ordered P937/P939 runs can supply direct pre/post TRUE-C values.

## 12. Results-language law

Use these labels exactly:

- **MEASURED:** physical checkpoint scored on the named instrument.
- **MEASURED DIAGNOSTIC:** physical experiment that isolates an effect but is not the target wire.
- **PROJECTED:** solver/pricing output without physical scoring.
- **ESTIMATE:** arithmetic/model inference with explicit basis and uncertainty.
- **PENDING:** dependency-gated work not yet measured.

Never put a pending value into a measured column. Never compare different cell populations without an explicit basis note.

## 13. Release checklist

Before publishing or merging, run from the project root:

```bash
cd tools/qtip2-backpack-campaign/wire-c-v2-2026-07-28
python3 code/verify_package.py
python3 code/recompute_results.py --check
python3 code/p908_direct_pricing.py
python3 code/verify_corrected_pricing.py
cd ../../..
python3 tools/publication_audit.py
```

Then verify repository manifests, run `git diff --check`, inspect every changed file, scan the full repository for private identities/paths/addresses/task identifiers, and review all source/public SHA pairs in `ARTIFACT_PROVENANCE.json`.
