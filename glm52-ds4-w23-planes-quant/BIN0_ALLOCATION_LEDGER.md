# BIN 0 allocation ledger

Updated 2026-07-19. BIN 0 changes **which units occupy which exact quarter-bpw tiers**. It does not train the fixed artifact's continuous parameters as a behavioral repair; that work belongs to BIN T.

Every decision below uses exact-hard forward semantics, an exact byte-neutral projection/replay, train-only selection, and one frozen final held-out read when the arm reaches that stage. Positive train loss is diagnostic only.

## Gradient/STE family: closed after eight held-out negatives

The representative L003 mechanism was real: entropy-floor/byte-dual/temperature-cosine training lowered matched train loss by 21.676%, and a split-k8192 arm closed the soft-to-hard gap to zero while reaching hard loss 0.014121 versus 0.024331 for its matched fixed control. Those mechanism wins did not transfer reliably to the 43-layer held-out surface.

The closure set includes frozen train-selected gates, two-sided de-starvation, full-43 promotion replicas, train-selected hard projection, compressed hard-STE, no-decay/hot-LR, current-window direction, and a selected-branch total-32 continuation. The eighth clean negative was the long-repair verdict below. Further cosmetic gradient-gate variants are retired.

### Fair no-decay/hot-LR test

The runtime optimizer audit proved that the real L003 gate was already in a unique Adam group with `weight_decay=0`. Finite 1x/8x/43x checks showed that LR 2.15 could decisively move the gate. The 10-step arm selected the exact k2048/k8192 swap (`p_swap=0.970045`, two moved units), but Lane-7 mean was 0.0681456521:

- versus exact step0 0.0675582543: **-0.869468%**, paired bootstrap 95% CI [-2.284724%, +0.530590%];
- versus matched fixed10 0.0686811237: **+0.779649%**, CI [-0.701230%, +3.136258%].

Verdict: gate movement and weight decay were not the remaining problem. Result SHA-256 `8373a216e9ddb255b391f6d74efea90635434728cb32eca8b70d4088f92cf4d6`; optimizer audit `77fea199173820642ddf1543e785d1d06b52edabd69e1a91fc7418da572630e4`.

### Current-window causal direction

At each predeclared gate boundary, the arm used the current train microbatch's exact hard swap-minus-incumbent contrast. Both boundaries retained the incumbent mechanically, but Lane-7 mean regressed to 0.0687266772:

- **-1.729504%** versus exact step0, CI [-3.553441%, -0.452640%];
- -0.066326% versus fixed10, CI [-1.040917%, +1.322271%];
- -0.852622% versus the no-decay predecessor, CI [-2.188466%, +0.070897%].

Verdict: direction correction alone is a sealed negative. Result SHA-256 `419428416e0a8a167576a6dc05e3cb5bb2030aa48f142a3fb1f71f864fd07102`.

### Selected-branch long repair: A32/B32

The selected exact-byte swap and same-seed/same-order fixed incumbent both received 32 matched doses without held-out tuning. Final means were:

| arm | Lane-7 mean |
|---|---:|
| exact step0 | 0.0675582543 |
| matched fixed A32 | 0.0691785850 |
| selected swap B32 | 0.0696811189 |

B32 was **0.726430% worse pooled** than A32 (paired-mean -0.760943%, CI [-2.115372%, +0.570045%]) and 3.142273% worse pooled than step0. B32 is rejected. The exact artifact remained 101,360,840,912 bytes; final seal SHA-256 `773ba3858c07c2f0f1575101579b7201e6ed115b19ba9ebd78e20a26dc97f673`.

## First positive signs

### Boundary-refresh representative

Three fixed seeds recomputed the exact hard direction at every one of ten train boundaries. Seeds were 69103, 69109, and 69121; seed 69109 was selected using train data only.

- selected seed: **+0.063656%**, CI [-0.019013%, +0.138154%];
- pooled three-seed mean: **+0.029719%**, hierarchical CI [-0.015061%, +0.085242%].

This was the first positive-sign held-out result, but both intervals cross zero. All six learned/control projections replayed at 13,750,272 bytes with zero delta. Result SHA-256 `fb0357febe5be84b2d610d699f6be07b18e71554c7b84813e43b2efe03b30ca6`.

### Class-tail-weighted arm

A 10-step train-frozen objective protected the top 20% of positions and 25% of windows with a normalized 4x tail weight while assigning 41.67% gradient mass to code and 41.67% to reasoning. The final held-out mean was 0.0675195425, p95 0.3075323, and p99 1.0603443.

- versus matched fixed10: **+1.691267%**, 95% CI **[+0.667027%, +3.176164%]**;
- versus the current-window causal arm: **+1.756428%**, CI **[+0.373760%, +3.205212%]**;
- versus exact step0/Combo-V4: +0.057301%, CI [-2.025803%, +2.101671%].

This is the first comparison with a wholly positive interval, but it is **not promotable**: the final eight-window panel contained zero reasoning windows, and the comparison against the actual start still crosses zero. Final seal SHA-256 `dccccbd9921421ac757d894dfa8f176771fb7c3b8900112cfe945cab0b787e23`; result SHA-256 `25679724755fbc1d7424ee315a3d0b15b045302e6b37f9204ab31869cacbcf51`.

### Other distinct arms

- Gumbel-ST, three seeds 69131/69143/69157, temperature 2.5→0.25: all arms hardened to the incumbent; selected seed lost 0.586019% to fixed, CI [-1.517630%, +0.530092%].
- Four-tier wide menu: hardened to four k4096 choices with zero moves; +0.015856%, CI [-1.233038%, +0.914262%].
- CVaR/top-5% arm: mean improved 0.4413% and p99 improved 5.1048%, while p95 regressed 1.2122%; paired mean interval crossed zero. This is a mixed distributional movement, not a promotion.

## ARM-M / ARM-F redesign

The gradient family is replaced by direct measurement.

### ARM-M: measured-delta discrete optimization

For each exact byte-neutral candidate pair:

1. measure hard-forward candidate-minus-incumbent ΔKL on at least 256 train-bank windows;
2. require mean ΔKL < 0 and improvement magnitude > 2× paired standard error;
3. require the same sign on a source-disjoint internal holdout;
4. keep Lane-7 sealed until a candidate passes the internal gate;
5. physically recount and replay the exact artifact bytes.

The first full-43 candidate used 258 measurements (129 discovery + 129 confirmation). The e192 demotion/e85 promotion was rejected: pooled candidate-minus-incumbent ΔKL was +0.000139440 with SE 0.000108815. Exact replay stayed 101,360,840,912 bytes; final seal `07336ad27614e5e1643c7d208b0d6bfb571973c03687658b769ad4ff27004afd`.

### ARM-F: forward/Fisher shortlist

ARM-F ranks per-unit adjacent-tier choices by measured or Fisher-weighted loss per byte, then sends a small exact candidate set to ARM-M. A compact pilot ranked 690 rows over 256 train-bank windows and measured 12 byte-neutral pairs. Its first passing discovery row demoted projection-2 e194 and promoted e118:

- discovery improvement +0.002461646684, SE 0.001190319192, z=2.068056, N=256;
- internal holdout improvement +0.002253496787, SE 0.002622374333, N=64, same sign but not independently significant;
- incumbent and selected wires both 6,934,528 bytes.

This is a compact-surface candidate, not a full-model or Lane-7 result. Result SHA-256 `7147ad79156cac954bc4172b8d8f6e9a032e4a251c14abb7b3c08a2a062683be`.

### Starting point and search policy

Use BAQ's equal-marginal-loss solution as the initial allocation, not greedy movement from the incumbent. Refine that solution with measured ΔKL, 2×SE gates, and internal holdout. If coordinate descent plateaus, search a small evolutionary up/down neighborhood while preserving exact bytes.

## Research anchors

- [BAQ: Efficient Bit Allocation Quantization for Large Language Models](https://arxiv.org/abs/2506.05664): Hessian-informed loss/bitwidth model, convex allocation, and closed-form equal-loss structure.
- [A KL Lens on Quantization](https://arxiv.org/abs/2604.13440): forward-only KL sensitivity for mixed precision; closest published analogue to ARM-F.
- [SliM-LLM](https://arxiv.org/abs/2405.14917): group-wise salience-driven bit allocation.
- UDP, *Up-or-Down Precision Quantization for Large Language Models via Evolutionary Search*: evolutionary search around a mixed-precision baseline.
- [Quant-dLLM](https://arxiv.org/abs/2510.03274): adaptive blockwise mixed precision at strict low-bit budgets.
- [Quantization Inflates Reasoning](https://arxiv.org/abs/2606.25519): reason to keep behavioral repair separate from allocation.
- [Cliff Tokens](https://arxiv.org/abs/2606.25524): motivates token-wise failure localization and statistical thresholds rather than raw tail mass alone.

## Reproduction contract

The model-specific ARM-F/ARM-M evaluator is not redistributed. Public receipts can still be schema-checked and hash-bound without inventing a runner CLI:

```bash
python3 -m json.tool "$INPUT/ARM_M_MEASURED.json" >/dev/null
shasum -a 256 "$INPUT/ARM_M_MEASURED.json" "$INPUT/EXACT_BYTE_LEDGER.json"
```

A complete reproduction must provide candidate tensors, banked teacher rows, the exact-byte serializer, at least 256 discovery windows, a source-disjoint confirmation set, and the final-only held-out rule described above.
