# Canonical end-to-end TRUE-C recipe

This is the single deterministic recipe for rebuilding and measuring the terminal pre-repair TRUE-C candidate. It intentionally separates three identities:

1. `f521-T physical TRUE-C` — the candidate measured by the P951 receipt.
2. `corrected-pricing V3` — the different assignment projected by the P931 solve.
3. `IQ4` — a different cell population retained as a reference.

Never combine their numbers into one implied candidate.

## 0. Verify the publication before using it

From this directory:

```bash
python3 code/verify_corrected_pricing.py
python3 code/verify_package.py
python3 -m py_compile \
  code/p929_run_true_c_refit.public.py \
  code/p929_true_c_overlay_adapter.public.py \
  code/p951_true_c_balanced64.public.py \
  code/p951_true_c_overlay_adapter.public.py \
  evaluation/toolkit/*.py
```

`*.public.py` files preserve the mission control flow but replace private task, path, host, and address strings. Before execution, bind `$SOURCE_ROOT`, `$PUBLIC_WORKSPACE`, and public node placeholders to the local sealed artifact store. The sealed pre-substitution SHA is recorded in `PACKAGE_MANIFEST.json`; the shipped-byte SHA is checked by `verify_package.py`.

## Frozen authority pins

| Authority | SHA-256 |
|---|---|
| BALANCED64_V1 | `7f756b898aea80cb4dd9320da4cd0c855f258d055f62ef6c37151d27857fa0ad` |
| Base assignment | `c9fb72e2bf7416ef48f33df229f9a3b5b5dd4f9e9b35a610d83fb1c49f4a050d` |
| Base physical wire manifest | `c24a1c0568a00fcb8460d7edfb7630187ef10c98e9d0c25c87aa0bccb1d89755` |
| Frozen f521 assignment | `f521cf07e0dce3c39739c7493b6eda82cd78d6b1566fadb2101691321566ca39` |
| Frozen f521 semantic map | `786b01a3f8c0197407e0025c80ca92c29b347a9c18de4b1ca48b7cf52ae08df6` |
| Physical 21,472-row active overlay | `9a4b709851c62c32f59b17556ef14d53e89cbbfc0fcc93686fc51530e4cf4d62` |
| Active-row canonical set | `7f21caeb50401f931c86d443b1538c9deb400c69dc41fb43aff5250e488a504e` |
| Changed-cell identity set | `01ff3e72338f007592ad5b83b687a0e1e875c39a749246b7790eac294c132327` |
| TRUE-C build identity | `13d1f887f8e6055f1f579730c2cc37be1e6c0754dd02256cf35a3a9f8c2d0a2f` |
| TRUE-C refit delta manifest | `6d13b82d49c49c55c4215b662cad4c488a1b8c81fb39a32e03096562ba604dc6` |
| P943 terminal refit seal | `90e6d6b131d14b353be2976848dc90e947cb6fc1cda376e03b760a63dce8d31c` |
| P951 measured receipt | `25dc5d5965b6e0e6c11db69ae05b7d64ec158dd41698e84e551db35270f1e5f7` |
| P951 output tensor set | `3529d33893a12d92dda96beba29c1a0e21adec6d008f2b32ced7d0066662c451` |
| P951 instrument identity | `c71b24d8c94927661d3aecd8899d59f0c825c9e7cd362b509372f202e2d31d50` |

The corrected V3 planning chain is pinned separately:

| V3 authority | SHA-256 |
|---|---|
| Corrected pricing | `c8673867b0fb7626232721d4939a9fdf95ef6d1a3de69698fd2a3d42398606c0` |
| Corrected vertical grid | `49407ff0114c5bcf9f7a68fbfc2a4822fee1839852aff5d89b8ce12d1251c203` |
| Calibration report | `6213107d728ac0df48be7121a082a6efa6f894d30c800e8db94315589c86a0d9` |
| Calibration validator | `9666d979b79ec576f55a4ea685bb1311b29910875fc227a24f470370e516b379` |
| P922 selection | `e776c293be491f080a630f7ba1d066ea0cc420c773be6758de2b4c92a3fb9818` |
| P928 interaction assignment | `62c26b9ea8f53aa2a2be84ff55b0e444100625f900832e096624ea178d9f9122` |
| P931 definitive independent verification | `60e6573f717e78fa8039a64938d7e444661e6583bd2c1549aa15906fcba4703a` |
| P931 reviewed source-artifact manifest | `d13db9c39f2da6620c432ba75ec1a5c45b1852b766418cc2e8d8a2b09e9e312a` |

## 1. Build the uniform QTIP2 anchor

The uniform anchor is a calibration/control rail, not the final mixed assignment.

Exact code surface:

- `../builders/qtip-rep16/qtip_wire_build_v3.py`
- `../builders/qtip-rep16/qtip_wire_build_serial_v2.py`
- `../builders/qtip-rep16/rate_contract.py`
- `../builders/qtip-rep16/rate_batched_ldlq.py`
- `../builders/qtip-rep16/triton_viterbi*.py`
- `../builders/qtip-rep16/verify_outputs.py`
- `../rail/p671/run_qtip_anchor.py`
- `artifacts/P880_QTIP2_ASSEALED_BALANCED64.public.json`

Required invariants:

1. Exact assignment and source-model hashes are checked before GPU allocation.
2. Every unit is built and read back by expected size and SHA.
3. Unit order follows the serialized identity manifest; never infer order from directory listing or archive position.
4. The BALANCED64 manifest is frozen before scoring.
5. The resulting QTIP2 control is labeled `MEASURED_REFERENCE`, not TRUE-C.

## 2. Replay the frozen mixed assignment

Exact code/config surface:

- `artifacts/WIRE_C_F521_BUILD_ASSIGNMENT.public.json`
- `artifacts/WIRE_C_F949_ASSIGNMENT.public.json`
- `artifacts/BASELINE_R_PHYSICAL_MANIFEST.public.json`
- `../builders/wire/stage_qtip_selected.py`
- `../builders/wire/canonical_shared_builder.py`
- `../builders/wire/build_overlay_shard.py`
- `code/p874_checkpoint_patch.py`

Procedure:

1. Hash the base assignment, f521 assignment, semantic map, physical base manifest, and each payload/codebook object.
2. Flatten exactly 43 layers × 256 experts × two projections = 22,016 identities.
3. Build or stage exactly the 21,472 identities where f521 differs from the immutable base; retain 544 base identities unchanged.
4. Resolve payloads by expected SHA from an immutable object index. A receipt path is provenance only.
5. Copy into a task-local partial tree, verify every byte count and SHA, then atomically rename.
6. Seal the ordered active-row canonical hash and identity-set hash shown above.

Do not reuse producer task IDs as authority. Semantic identity plus payload/codebook hashes are the authority.

## 3. Merge recovered QTIP anchors

Exact recovery code:

- `code/rebuild_anchor_worker.public.py`
- `code/restore_anchor_worker.public.py`
- `code/bulk_transfer.py`
- `doctrine/KASA_WEDGE_DOCTRINE.md`

Procedure:

1. Reconstruct only the missing identity prefix from a verified bank/manifest.
2. Rehash both the banked pair and destination pair before adoption.
3. Use exact-CAS claim transfer; preserve the prior claim preimage in the receipt.
4. Resume from the newest complete layer/codebook boundary. Never recompute a sealed prefix after a verified restart.
5. A path rename with unchanged object SHA must pass; wrong or duplicate SHA objects must fail before GPU work.

## 4. Exact weighted TRUE-C refit

Exact mission sources, privacy-substituted only:

- `code/p929_run_true_c_refit.public.py` — sealed source SHA `e621a936a3a5d118caa61f7964f97064bb8a7888f54dcbea9a59548f3d09683b`
- `code/p929_true_c_overlay_adapter.public.py` — sealed source SHA `2bf3e14c0dd3ef76413737981eeb8b62e37fe0e87afdc7114086793ac275428c`
- `artifacts/P943_TRUE_C_TERMINAL_SEAL.public.json`

Frozen method:

- target: every current VQ row whose active codebook SHA equals the frozen base codebook for the same layer/tier/projection;
- 2,860 rows, 80 shared codebooks, 25 layers;
- tiers: 173 `d4_k1024`, 1,374 `d4_k2048`, 1,313 `d4_k4096` rows;
- projections: 1,603 down, 1,257 fused13 rows;
- fit experts: 17, 77, 177;
- objective: scale-squared weighted k-means++ followed by 15 Lloyd iterations;
- scale grid: W3v2 e43 LUT with SSE offsets -4 through +2;
- assignment: nearest neighbor against the serialized fp16 codebook;
- seed: 0;
- corpus windows used for codebook fitting: no.

Every codebook is persisted before cells are emitted. Layer/codebook receipts are prefix-validating and quarantined fail-closed on drift. The terminal seal must close 80/80 codebooks and 2,860/2,860 rows, with every layer receipt pinned.

## 5. Terminal BALANCED64_V1 measurement

Exact mission sources, privacy-substituted only:

- `code/p951_true_c_balanced64.public.py` — sealed source SHA `131e34e283222b3ea1e8048971e8cb22009303a4ba903233f280d5b34ba37211`
- `code/p951_true_c_overlay_adapter.public.py` — sealed source SHA `cd51a54df68b935c72b7324935841b1bd6cbe48c0903f267d75af97becc1af37`
- `artifacts/BALANCED64_V1.json`
- `artifacts/P951_TRUE_C_BALANCED64.public.json`

Frozen geometry:

- direction: `KL(teacher||candidate)`;
- support: 8,192;
- cutoff: 1,024;
- windows: 64 exact IDs from BALANCED64_V1;
- positions: 65,536;
- attention: eager;
- microbatch: 2;
- chunk size: 64;
- full 43-layer physical coverage;
- exact 21,472/21,472 changed cells applied;
- zero substitution, zero quarantine, immutable base unmodified.

Terminal vector:

| class | mean KL | 95% window-mean CI |
|---|---:|---:|
| global | 0.06829414627618949 | [0.05417328304221277, 0.08241500951016621] |
| agentic | 0.07879656974187459 | [0.04338859086063873, 0.11420454862311044] |
| chat | 0.021183150884045005 | [0.015004703604817218, 0.027361598163272792] |
| code | 0.05501697946566645 | [0.04188852536003566, 0.06814543357129724] |
| multilingual | 0.11238435483229318 | [0.09641440183003669, 0.12835430783454968] |
| prose | 0.09759553403503682 | [0.07278204654519906, 0.12240902152487458] |
| reasoning | 0.014495197391988604 | [0.010310002889108829, 0.01868039189486838] |

The IQ4 reference is approximately 0.07204 global KL at the reported 4.06 bpw. The descriptive difference is -0.00374585372381051, but IQ4 is a different cell population and the comparison is not a paired same-candidate estimate.

## 6. Corrected V3 solve remains a separate planning result

Reproduce the projected solve with:

```bash
python3 code/verify_corrected_pricing.py
python3 code/solve_v2_reweighted.py --help
```

Inputs and outputs are pinned in `artifacts/P930_*`, `artifacts/P931_V3_DEFINITIVE.public.json`, and `code/V2_REWEIGHTED_CONFIG.public.json`. The historical `P931_V3_FIRST_FEASIBLE.public.json` value (`0.06913222309403669`) is lineage only.

The canonical P931 result is a projected reweighted objective of `0.035078633039490076`, exact size `101,346,700,382` bytes (`29` bytes slack), and relative gap `2.7554042339551337e-06`. It is a feasible time-limit incumbent, not proven optimal, not measured, and not a physical BALANCED64 score. The public JSON is a derived summary bound to reviewed evidence hashes; the private assignment payload is not redistributed (see `P958_ASSIGNMENT_RECOVERY_STATUS.md`). It does not replace, correct, or reinterpret the P951 f521-T measurement.

## 7. Exact-equal scorer acceleration

P963 changes the physical-read implementation without changing scorer semantics. Public code and detailed receipts are under `acceleration/`:

```bash
cd acceleration
bash code/launch_p963.sh
```

Bind the public path/host placeholders to the local immutable authority store before execution. The terminal gate requires the same output-set SHA (`3529d33893a12d92dda96beba29c1a0e21adec6d008f2b32ced7d0066662c451`), `64/64` bit-exact tensors, maximum absolute per-position delta `0.0`, and identical global/six-class means. Microbatch 4 failed the `<=1e-12` exactness gate, so microbatch 2 is binding. The measured result is `1495.7971739768982` seconds versus `3643.123103618622` seconds (`2.4355729286027437x`). This is an acceleration result, not a new model-quality result.

## 8. Downstream functional evaluation

Use `EVALUATION_PROTOCOL.md`, `evaluation/P967_INFERENCE_PROTOCOL.public.json`, `evaluation/P968_AUTHORITY_MAP.public.json`, and `evaluation/toolkit/`.

No functional paired TRUE-C-vs-IQ4 result is published here. The protocol is preregistered; historical harness/reference rows remain labeled as references.
