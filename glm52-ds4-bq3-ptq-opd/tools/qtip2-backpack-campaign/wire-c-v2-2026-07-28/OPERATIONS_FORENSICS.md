# Operational forensics and reusable lessons

This chronicle records what succeeded, what failed, and which receipts are authoritative. It is intentionally candid: content validity, transport-policy validity, candidate identity, and publication validity are separate axes.

## Executive timeline

1. Uniform QTIP controls established the scoring and decoder rails.
2. P841 optimized the proven scorer without changing decision-bearing math.
3. P877 demonstrated checkpoint/restart recovery, but also exposed tmpfs and transport-policy failures.
4. P922 measured the cost of substituting restored/base VQ codebooks; it was a diagnostic, not TRUE-C.
5. P928 measured mixed-tier interaction and repaired the affected vertical-price rows.
6. P930 sealed the corrected V3 pricing surface and calibration report.
7. P931 solved that corrected surface. Its definitive feasible incumbent is projected and belongs to a different candidate; the earlier first-feasible projection is superseded.
8. P929 regenerated the exact shared codebooks required by f521-T.
9. P943 sealed the 80-codebook/2,860-row TRUE-C artifact chain.
10. P951 independently measured the terminal f521-T candidate on BALANCED64_V1.
11. P963 replayed the P951 scorer with exact-equal outputs at 2.435573x lower wall time.
12. P936/P953 converted recurring receipt and path failures into structural negative tests. P957 canonicalization was still review-blocked at publication time and must not be reported as deployed.
13. P967/P968 preregistered the binding n=5 sampled and three-repeat greedy functional evaluation; no paired outcome is claimed here.

## P841: accelerate only after bit-exactness is banked

P841 started from a three-layer scorer already proven bit-exact against activation hashes, a 1 GiB logit stream, all 64 KLD rows, and the global mean. Profiling identified:

- exact forwards: 70.3 s;
- QTIP decode/materialization: 49.5 s, overlapped;
- post-forward readout: about 26.7 s.

The accepted patch reached 84.59689831733704 s for three layers, or 28.19896610577901 s/layer. It passed the <=30 s/layer bar but honestly missed the >=1.8× bar at 1.695411434140093×. A second patch was stopped and rolled back when MemAvailable fell 33,329,152 bytes below the 8 GiB floor.

Pins:

- final receipt: `8df5694b2693ee357b01b2f87babab8595c790636a0fdee3fc9224ae857c4762`;
- patch: `4fdcb1d67b93ab268fca54e4f3f7005dd927dd74a9405ddd80ae9c49f7cbaa9e`;
- exact release: `e7a246a5303fcbb34029d3ff1d6811a5f736c9f2e722dd0e7b3aac2199d1f9d4`.

Lesson: once correctness is banked, profile once and patch measured sinks only. Do not pay for another reference campaign or silently relax a resource floor to chase a speed target.

## P877: interrupted build, restart, and the transport caveat

P877 reused the proven scorer plus per-layer P874 checkpoints. The first run wrote checkpoints only under tmpfs. An external reboot erased the L000-L003 anchors and killed the process. Because no compatible persistent checkpoint survived, the operator explicitly authorized a restart from L000. The relaunched process was detached and verified from a second session.

The recovery then improved:

- every completed layer wrote a rolling checkpoint;
- L018 was copied to persistent storage with matching source/destination SHA;
- later persistent anchors were rotated forward;
- an exact L039 archive was staged directly after size/SHA verification.

Terminal content result:

- receipt: `8742b46a6727b8cebed9162702c91c840a9cf502b54b14bf1469f4a2fc69a57e`;
- final persistent L042 checkpoint: `b6688c9ceb55101c28baf8767f16c447946ef6549a255540568c1ef81baef871`;
- global KL: `0.06588095206672909`;
- exact QTIP3 coverage: 20,480/20,480 cells.

However, the receipt records online NAS sourcing for terminal layers, and an earlier online NAS fallback was directly observed after policy had forbidden that transport. Therefore the content is SHA-valid but the lane is `NONCONFORMANT_ONLINE_NAS`. The authoritative QTIP3 value was separately sealed by conformant lanes; P877 is forensic corroboration, not the sole authority.

Lessons:

1. tmpfs is scratch, never the only restart boundary;
2. detached launch plus second-session verification prevents session-fence deaths;
3. content hashes do not erase a transport-policy violation;
4. a superseded backup lane should stop at a sealed boundary rather than continue consuming scarce resources.

## P922: restored-codebook substitution is a diagnostic

P922 joined and substituted exactly 3,803 selected VQ identities. It measured a global penalty of `0.02925963216194956` KL, CI95 `[0.02357004720217415, 0.03494921712172497]`. The restored-VQ diagnostic candidate itself measured `0.1466261360478779` global KL.

This established that codebook identity materially changes quality. It did not directly measure TRUE-C and must never be relabeled as such. The later TRUE-C chain regenerated shared codebooks rather than carrying the substituted base codebooks forward.

Pinned inputs used downstream:

- exact P922 selection: `e776c293be491f080a630f7ba1d066ea0cc420c773be6758de2b4c92a3fb9818`;
- P922 BALANCED64 receipt: `c2604f6386e00c1ab38b238cd5940f85d1ab12ec20e18f43dfc9efcece29be8e`;
- terminal verification: `d3cbae01335d4fb3809275d72d9f6201c439f2598cd7c41ba0e57d81b36733c5`.

## P928: mixed-tier interaction must be measured, not assumed additive

The preregistered mixed pattern produced:

- measured global: `0.07943175833066395`;
- uncorrected predicted global: approximately `0.079353505787`;
- global interaction: `0.00007825254366380929`.

Interaction by class:

- agentic `-0.0009052421974351799`;
- chat `-0.00042987265425962345`;
- code `+0.005783938457895543`;
- multilingual `+0.002980289533620867`;
- prose `-0.005383297528387301`;
- reasoning `-0.000704459859670185`.

The global effect was tiny while the class effects were not. This is why global-only calibration can conceal a material code/prose redistribution.

Pins:

- interaction receipt: `858fde4ca2252212fb4f03919fe2bda0e0672b977d125ce2b98f3898ce26d096`;
- mixed assignment: `62c26b9ea8f53aa2a2be84ff55b0e444100625f900832e096624ea178d9f9122`;
- post-stage payload manifest: `5520e6686a9cf3bf3ecd741095d298779999835346c78c081d74db5f30c4c3c2`.

Lesson: apply the interaction once. P930 bakes it into the three affected V3 rows; P931 must not add it again.

## P930: corrected pricing with disclosed residual misses

P930 joined the exact P922 selection, applied the P928 interaction to the affected rows, and sealed a corrected nonnegative V3 pricing surface.

Pins:

- pricing: `c8673867b0fb7626232721d4939a9fdf95ef6d1a3de69698fd2a3d42398606c0`;
- grid: `49407ff0114c5bcf9f7a68fbfc2a4822fee1839852aff5d89b8ce12d1251c203`;
- report: `6213107d728ac0df48be7121a082a6efa6f894d30c800e8db94315589c86a0d9`;
- validator: `9666d979b79ec576f55a4ea685bb1311b29910875fc227a24f470370e516b379`.

All four calibration wires closed globally within 5%, but six per-class misses remained and are preserved in the report. A global pass is not permission to hide class failures.

## P931: definitive solve, projected candidate

P931 solved the corrected V3 surface under the 101,346,700,411-byte envelope. The canonical feasible time-limit incumbent is:

- projected reweighted objective `0.035078633039490076`;
- class projection: agentic `0.03958745712003608`, chat `0.008532822455696412`, code `0.05105645114446229`, multilingual `0.04864400347424909`, prose `0.0352992491602176`, reasoning `0.005687227334670042`;
- exact bytes `101,346,700,382`, slack `29`;
- best bound `0.03507853638367621`;
- relative gap `2.7554042339551337e-06`;
- `20,718` changed cells.

Independent verification receipt SHA: `60e6573f717e78fa8039a64938d7e444661e6583bd2c1549aa15906fcba4703a`; reviewed artifact-manifest SHA: `d13db9c39f2da6620c432ba75ec1a5c45b1852b766418cc2e8d8a2b09e9e312a`.

The historical first-feasible projection `0.06913222309403669` remains lineage only. The canonical result is still a model projection: it is not optimality-proven, not physically scored, and not the f521-T candidate measured by P951. Its public receipt is a derived summary because the private assignment payload was not durably redistributed; `P958_ASSIGNMENT_RECOVERY_STATUS.md` records the limitation.

## P929/P943: exact codebook regeneration and terminal seal

P929 targeted every active VQ row still bound to a frozen base codebook:

- 2,860 rows;
- 80 shared codebooks;
- 25 layers;
- 1,603 down and 1,257 fused13 rows.

The refit objective, seed, fit experts, and replay checks are frozen in `code/p929_run_true_c_refit.public.py`. P943 sealed the final ledger:

- terminal seal source SHA: `90e6d6b131d14b353be2976848dc90e947cb6fc1cda376e03b760a63dce8d31c`;
- active overlay: `9a4b709851c62c32f59b17556ef14d53e89cbbfc0fcc93686fc51530e4cf4d62`;
- build identity: `13d1f887f8e6055f1f579730c2cc37be1e6c0754dd02256cf35a3a9f8c2d0a2f`;
- delta manifest: `6d13b82d49c49c55c4215b662cad4c488a1b8c81fb39a32e03096562ba604dc6`.

The ledger is shipped in `artifacts/P943_TRUE_C_TERMINAL_SEAL.public.json` so every codebook receipt remains auditable.

## P951: terminal physical measurement

P951 ran the f521-T physical candidate independently through the parity-proven BALANCED64 rail:

- receipt: `25dc5d5965b6e0e6c11db69ae05b7d64ec158dd41698e84e551db35270f1e5f7`;
- output tensor set: `3529d33893a12d92dda96beba29c1a0e21adec6d008f2b32ced7d0066662c451`;
- instrument identity: `c71b24d8c94927661d3aecd8899d59f0c825c9e7cd362b509372f202e2d31d50`;
- coverage: 21,472/21,472 changed cells, all 43 layers;
- pack fraction: 1.0;
- zero substitution and zero quarantine.

The terminal global result is `0.06829414627618949`, with full vector and confidence intervals in `artifacts/P951_TRUE_C_BALANCED64.public.json`.

## P963: profile first, accelerate without changing bytes

P963 targeted the measured P951 sinks: layer preparation (3,602.6 s), prefetch wait (3,010.2 s), and physical overlay fill (834.6 s). These counters overlap and are not additive wall-time partitions. The accepted implementation batched expected-SHA-namespaced peer staging, overlapped it with immutable-base fill, double-buffered the next layer, skipped redundant base fill for exact full-overlay layers, retained mmap for partial layers, and kept microbatch 2 after microbatch 4 failed the `<=1e-12` exactness gate.

The accelerated read took `1495.7971739768982` s versus `3643.123103618622` s baseline (`2.4355729286027437x`, `58.94189871071991%` wall reduction). The raw output-set SHA remained `3529d33893a12d92dda96beba29c1a0e21adec6d008f2b32ced7d0066662c451`; all 64 tensors were bit-exact and maximum per-position delta was `0.0`. The public code and detailed receipts are under `acceleration/`.

## P936/P953/P957: make recurring failures structural

P936 addressed receipt and identity ambiguity with four structural controls shipped under `structural-guards/p936/`: an append-only `store/<sha256>.bin` authority store; an exact-schema substitution waiver that requires a byte-present measured receipt and at least 64 windows; a reverse protected-SHA index that rejects reclaim unless an `ARCHIVE_FIRST` receipt proves exact external readback; and a seal-time dependency census requiring byte-exact copies on at least two distinct hosts. Positive and negative tests exercise missing/wrong SHA, estimate-only waiver, protected reclaim, archive readback, and one-copy sealing.

P953 then addressed the historical path-authority bug and layer resume:

- provenance path becomes metadata only;
- expected SHA resolves through an immutable authority index;
- missing, duplicate, and wrong bytes fail closed;
- inherited prefixes bind semantic identity plus payload/codebook hashes;
- completed layers resume at the first unfinished layer;
- regression coverage includes immutable-index self-pinning, duplicate/missing/wrong-byte rejection, semantic inherited-prefix binding, and exact contiguous resume.

The P953 patch was sealed ready-to-adopt but was not deployed into a healthy scorer.

P957 attempted canonical integration. Independent review found unresolved blockers at publication time:

1. the production reader did not preserve a preseeded completed-layer history;
2. receipt history could be dropped when a new layer was written;
3. object resolution occurred after GPU-backed initialization rather than before GPU touch;
4. tests exercised planning lists rather than the full production reader/builder path;
5. a referenced deploy manifest was absent;
6. the documented default test command failed on the stated older Python runtime.

Therefore this package records the guard contract but does not claim P957 canonical deployment. Future measurement cards must require an expected-SHA manifest and must prove, through the production entrypoint, that wrong/duplicate bytes abort before GPU and that a sealed L000-L013 prefix resumes at L014 without recomputation.

## Reproducibility laws distilled from the campaign

1. Candidate identity is assignment + codebook/payload hashes + physical coverage, not a label or path.
2. A receipt path is provenance; immutable object SHA is authority.
3. Source SHA and public-copy SHA are different when privacy substitution occurs; publish both.
4. Checkpoint state must be persistent, hashed, and bound to the exact completed-layer receipt set.
5. Content validity, transport-policy validity, and operational conformance are separate labels.
6. Global calibration cannot erase class-level misses.
7. Mixed-tier interaction is measured once and applied once.
8. A projected solver vector and a physical measurement remain separate even if their numbers are close.
9. Historical harness rows are references until the exact candidate is served and scored.
10. No result is terminal without exact release, independent reread, and a machine-readable receipt.
