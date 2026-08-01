#!/usr/bin/env python3
"""One-layer QTIP/backpack SOLVER phase profile.

The scientific work is a direct replay of SOLVER_PRICING_V2 exact_per_cell_prices:
all 256 experts, both projections, all eight sealed tiers, repaired/base variants,
certified d4/k2048 shortlist with exact fallback, and the 32/64-window capture law.
Only phase timing and compact fsynced profile rows are added; pricing artifacts are
not mutated.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Callable

import torch

BUCKETS = (
    "codebook_distance_sweeps",
    "dequant_proxy_gemvs",
    "hessian_proxy_reads",
    "host_staging",
    "python_dispatch",
)


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    dfd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


def append_fsync(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(obj, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def assignment_sha256(assignment_payload: dict[str, dict[str, Any]]) -> str:
    """Canonical exact assignment digest used by public solve/evaluate receipts."""
    import hashlib

    return hashlib.sha256(
        json.dumps(assignment_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class Recorder:
    def __init__(self, started_mono: float, sample_s: float) -> None:
        self.started_mono = started_mono
        self.sample_s = sample_s
        self.totals = {k: 0.0 for k in BUCKETS}
        self.subphases: dict[str, dict[str, Any]] = {}
        self.async_work_totals = {k: 0.0 for k in BUCKETS}
        self.async_subphases: dict[str, dict[str, Any]] = {}
        self.windows: dict[int, dict[str, float]] = {}

    def snapshot(self) -> dict[str, float]:
        return dict(self.totals)

    def _allocate_window(self, bucket: str, amount: float, t0: float, t1: float) -> None:
        if amount <= 0:
            return
        span = max(t1 - t0, 1e-12)
        first = max(0, int((t0 - self.started_mono) // self.sample_s))
        last = max(first, int(max(0.0, t1 - self.started_mono - 1e-12) // self.sample_s))
        for wid in range(first, last + 1):
            w0 = self.started_mono + wid * self.sample_s
            w1 = w0 + self.sample_s
            overlap = max(0.0, min(t1, w1) - max(t0, w0))
            if overlap <= 0:
                continue
            row = self.windows.setdefault(wid, {k: 0.0 for k in BUCKETS})
            row[bucket] += amount * overlap / span

    def add(self, bucket: str, subphase: str, amount: float, t0: float, t1: float) -> None:
        amount = max(0.0, float(amount))
        self.totals[bucket] += amount
        row = self.subphases.setdefault(subphase, {"bucket": bucket, "seconds": 0.0, "calls": 0})
        if row["bucket"] != bucket:
            raise RuntimeError((subphase, row["bucket"], bucket))
        row["seconds"] += amount
        row["calls"] += 1
        self._allocate_window(bucket, amount, t0, t1)

    def wall(self, bucket: str, subphase: str, fn: Callable[[], Any], sync_cuda: bool = False) -> Any:
        if sync_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn()
        if sync_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        self.add(bucket, subphase, t1 - t0, t0, t1)
        return out

    def gpu(self, bucket: str, subphase: str, fn: Callable[[], Any]) -> Any:
        # Synchronize only the measured/default stream. A device-wide sync here
        # would serialize the dedicated next-weight prefetch stream and defeat
        # the exact solver's double buffer.
        torch.cuda.current_stream().synchronize()
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter()
        e0.record()
        out = fn()
        e1.record()
        e1.synchronize()
        t1 = time.perf_counter()
        device_s = max(0.0, float(e0.elapsed_time(e1)) / 1000.0)
        wall_s = max(0.0, t1 - t0)
        # Rare timer quantization can make event time microscopically exceed host wall.
        device_s = min(device_s, wall_s)
        self.add(bucket, subphase, device_s, t0, t1)
        dispatch_s = max(0.0, wall_s - device_s)
        if dispatch_s:
            self.add("python_dispatch", f"dispatch::{subphase}", dispatch_s, t0, t1)
        return out

    def add_untracked(self, subphase: str, amount: float, t0: float, t1: float) -> None:
        self.add("python_dispatch", subphase, max(0.0, amount), t0, t1)

    def add_async_work(
        self,
        bucket: str,
        subphase: str,
        amount: float,
        queued_mono: float,
        completed_mono: float,
    ) -> None:
        """Report overlapped device work without double-counting exclusive wall."""
        amount = max(0.0, float(amount))
        self.async_work_totals[bucket] += amount
        row = self.async_subphases.setdefault(
            subphase,
            {
                "bucket": bucket,
                "device_work_seconds": 0.0,
                "calls": 0,
                "queue_to_complete_wall_seconds": 0.0,
            },
        )
        if row["bucket"] != bucket:
            raise RuntimeError((subphase, row["bucket"], bucket))
        row["device_work_seconds"] += amount
        row["calls"] += 1
        row["queue_to_complete_wall_seconds"] += max(
            0.0, completed_mono - queued_mono
        )


def main(
    argv: list[str] | None = None,
    *,
    emit_summary: bool = True,
) -> dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="task-owned mission root")
    ap.add_argument("--source-root", type=Path, required=True, help="read-only SOLVER_PRICING_V2 root")
    ap.add_argument(
        "--model-root",
        type=Path,
        required=True,
        help="local immutable model root containing the 0731 checkpoint",
    )
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--windows", type=int, default=32)
    ap.add_argument("--sample-seconds", type=float, default=90.0)
    ap.add_argument(
        "--tiers",
        default=",".join(("d4_k2048", "d4_k4096")),
        help="comma-separated sealed VQ tiers to score",
    )
    ap.add_argument(
        "--staging-root",
        type=Path,
        help="optional resident immutable plane/weight staging root",
    )
    ap.add_argument(
        "--capture-root",
        type=Path,
        help="manifest-bound public capture bank for the selected layer",
    )
    ap.add_argument(
        "--implementation",
        choices=("serial", "optimized", "exact-gemm"),
        default="exact-gemm",
    )
    ap.add_argument(
        "--audit-codeword-assignments",
        action="store_true",
        help="hash every exact codeword winner for accelerated/reference parity audits",
    )
    args = ap.parse_args(argv)

    from . import solver_core as core

    selected_tiers = tuple(
        tier.strip() for tier in args.tiers.split(",") if tier.strip()
    )
    if not selected_tiers or len(set(selected_tiers)) != len(selected_tiers):
        raise ValueError(f"invalid tier selection: {args.tiers!r}")
    unknown_tiers = [tier for tier in selected_tiers if tier not in core.TIERS]
    if unknown_tiers:
        raise ValueError(f"unknown tiers: {unknown_tiers}")

    # Materialize selected codebook catalogs into the task-owned tier root.
    # Capture banks are consumed only through the manifest-bound --capture-root.
    args.root.mkdir(parents=True, exist_ok=True)
    core.stage_codebooks(args.root, args.source_root, selected_tiers, args.layer)

    out = args.root / "profile" / f"L{args.layer:03d}"
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / "PROFILE_ROWS.jsonl"
    if rows_path.exists():
        rows_path.unlink()

    torch.set_grad_enabled(False)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    epoch_started = time.time()
    mono_started = time.perf_counter()
    rec = Recorder(mono_started, args.sample_seconds)
    layer = args.layer
    index = core.checkpoint_index(args.model_root)
    layer_map = core.weight_shard_map(index, layer)

    # Source/hash validation is pure host I/O. Start both immutable input gates
    # together and overlap them with capture/H construction; consumers still
    # fail closed on the futures before touching either staged input family.
    def timed_stage(fn: Callable[[], Any]) -> tuple[Any, float, float]:
        t0 = time.perf_counter()
        value = fn()
        return value, t0, time.perf_counter()

    stage_pool = ThreadPoolExecutor(max_workers=2)
    plane_stage_future = stage_pool.submit(
        timed_stage,
        lambda: core.plane_paths(
            args.root,
            layer,
            selected_tiers=selected_tiers,
            staging_root=args.staging_root,
        ),
    )
    weight_stage_future = stage_pool.submit(
        timed_stage,
        lambda: core.open_shards(
            args.root,
            layer,
            layer_map,
            model_root=args.model_root,
            staging_root=args.staging_root,
        ),
    )

    # H proxy: receipt validation + mmap reads + exact fused-diagonal construction.
    captures = rec.wall(
        "hessian_proxy_reads", "startup.capture_load",
        lambda: core.load_captures(
            args.root,
            layer,
            args.windows,
            staging_root=args.staging_root,
            capture_root=args.capture_root,
        ),
    )
    try:
        hfused, counts = rec.wall(
            "hessian_proxy_reads", "startup.hdiag_fused",
            lambda: core.hdiag_fused(captures),
        )
    except RuntimeError as exc:
        if "unrouted experts" not in str(exc):
            raise
        captures = rec.wall(
            "hessian_proxy_reads", "startup.capture_load_64_fallback",
            lambda: core.load_captures(
                args.root,
                layer,
                64,
                staging_root=args.staging_root,
                capture_root=args.capture_root,
            ),
        )
        hfused, counts = rec.wall(
            "hessian_proxy_reads", "startup.hdiag_fused_64_fallback",
            lambda: core.hdiag_fused(captures),
        )
    actual_windows = len(captures)

    # Standard production staging: source planes and the full preview weight shard
    # are pulled exactly as the original solver does; no warm-start/cached prices.
    def resolve_plane_stage() -> dict[str, Path]:
        value, t0, t1 = plane_stage_future.result()
        rec.add_async_work("host_staging", "startup.plane_stage", t1 - t0, t0, t1)
        return value

    paths = rec.wall(
        "host_staging", "startup.plane_stage_wait", resolve_plane_stage
    )
    plane_data = rec.wall(
        "host_staging", "startup.plane_mmap",
        lambda: {tier: torch.load(path, map_location="cpu", weights_only=True, mmap=True)
                 for tier, path in paths.items()},
    )

    cb: dict[tuple[str, str, str], torch.Tensor] = {}
    cb_md5: dict[tuple[str, str, str], str] = {}

    def load_codebooks() -> None:
        for tier in selected_tiers:
            base_obj = torch.load(
                core.catalog_path(args.root, "base", tier, layer),
                map_location="cpu", weights_only=True,
            )
            for variant in core.variants(tier):
                source_variant = "repaired" if variant == "deployed" else "base"
                obj = (
                    torch.load(core.catalog_path(args.root, source_variant, tier, layer),
                               map_location="cpu", weights_only=True)
                    if source_variant == "repaired" else base_obj
                )
                for projection in ("13", "2"):
                    tensor = obj[f"cb{projection}"].to(core.DEVICE).float()
                    cb[(tier, variant, projection)] = tensor
                    cb_md5[(tier, variant, projection)] = core.tensor_md5(obj[f"cb{projection}"])

    rec.wall("host_staging", "startup.codebook_load", load_codebooks, sync_cuda=True)

    certified_shortlists: dict[
        tuple[str, str], tuple[torch.Tensor, torch.Tensor]
    ] = {}
    resident_candidates: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    exact_codebook_plans: dict[tuple[str, str], Any] = {}
    exact_search_tiers = {
        tier for tier in selected_tiers if tier in {"d4_k2048", "d4_k4096"}
    }
    for projection in ("13", "2"):
        if args.implementation == "optimized":
            resident_candidates[projection] = rec.gpu(
                "codebook_distance_sweeps", f"startup.resident_candidates.P{projection}",
                lambda projection=projection: core.build_resident_candidate_slab(
                    cb[("d4_k4096", "base", projection)],
                    cb[("d4_k2048", "base", projection)],
                ),
            )
        elif args.implementation == "exact-gemm":
            for tier in sorted(exact_search_tiers):
                exact_codebook_plans[(tier, projection)] = rec.gpu(
                    "codebook_distance_sweeps",
                    f"startup.exact_codebook_plan.{tier}.P{projection}",
                    lambda projection=projection, tier=tier: core.prepare_exact_codebook_plan(
                        cb[(tier, "base", projection)]
                    ),
                )
        elif args.implementation == "serial":
            for tier in sorted(exact_search_tiers):
                certified_shortlists[(tier, projection)] = rec.gpu(
                    "codebook_distance_sweeps",
                    f"startup.shortlist.{tier}.P{projection}",
                    lambda projection=projection, tier=tier: core.build_certified_shortlist(
                        cb[("d4_k4096", "base", projection)],
                        cb[(tier, "base", projection)],
                    ),
                )

    def resolve_weight_stage() -> dict[str, Any]:
        value, t0, t1 = weight_stage_future.result()
        rec.add_async_work(
            "host_staging", "startup.weight_shard_stage", t1 - t0, t0, t1
        )
        return value

    handles = rec.wall(
        "host_staging", "startup.weight_shard_stage_wait", resolve_weight_stage
    )
    stage_pool.shutdown(wait=True)

    row_count = 0
    fallback_vectors = 0
    certified_vectors = 0
    approximate_vectors = 0
    exact_vectors = 0
    scientific_rows: list[dict[str, Any]] = []
    expert_batch_size = 4 if args.implementation in {"optimized", "exact-gemm"} else 1
    expert_batches = [
        list(range(start, min(256, start + expert_batch_size)))
        for start in range(0, 256, expert_batch_size)
    ]
    prefetch_stream = torch.cuda.Stream() if args.implementation == "exact-gemm" else None
    prefetched = None
    if prefetch_stream is not None:
        first_staged = rec.wall(
            "host_staging", "batch.weight_cpu_stage",
            lambda: core.stage_weights_batch_cpu(
                handles, layer_map, layer, expert_batches[0]
            ),
        )
        prefetched = rec.wall(
            "host_staging", "batch.weight_prefetch_submit",
            lambda: core.prefetch_staged_weights(first_staged, prefetch_stream),
        )

    for batch_number, expert_ids in enumerate(expert_batches):
        if args.implementation == "exact-gemm":
            assert prefetched is not None and prefetch_stream is not None
            fused_batch, down_batch, prefetch_timing = rec.wall(
                "host_staging", "batch.weight_prefetch_wait",
                lambda prefetched=prefetched: core.wait_prefetched_weights(prefetched),
            )
            rec.add_async_work(
                "host_staging",
                "batch.weight_h2d_dequant_async",
                prefetch_timing["device_seconds"],
                prefetch_timing["queued_mono"],
                prefetch_timing["completed_mono"],
            )
            next_prefetched = None
            if batch_number + 1 < len(expert_batches):
                next_ids = expert_batches[batch_number + 1]
                next_staged = rec.wall(
                    "host_staging", "batch.weight_cpu_stage",
                    lambda next_ids=next_ids: core.stage_weights_batch_cpu(
                        handles, layer_map, layer, next_ids
                    ),
                )
                next_prefetched = rec.wall(
                    "host_staging", "batch.weight_prefetch_submit",
                    lambda next_staged=next_staged: core.prefetch_staged_weights(
                        next_staged, prefetch_stream
                    ),
                )
        elif args.implementation == "optimized":
            fused_batch, down_batch = rec.wall(
                "host_staging", "batch.weight_materialize",
                lambda expert_ids=expert_ids: core.load_weights_batch(
                    handles, layer_map, layer, expert_ids
                ),
                sync_cuda=True,
            )
        else:
            fused_one, down_one = rec.wall(
                "host_staging", "expert.weight_materialize",
                lambda expert=expert_ids[0]: core.load_weights(
                    handles, layer_map, layer, expert
                ),
                sync_cuda=True,
            )
            fused_batch, down_batch = fused_one.unsqueeze(0), down_one.unsqueeze(0)

        for batch_index, expert in enumerate(expert_ids):
            expert_t0 = time.perf_counter()
            before = rec.snapshot()
            fused = fused_batch[batch_index]
            down = down_batch[batch_index]
            x = rec.wall(
                "hessian_proxy_reads", "expert.capture_route_read",
                lambda expert=expert: core.expert_x(captures, expert),
            )
            hdown = rec.gpu(
                "dequant_proxy_gemvs", "expert.down_hdiag_gemms",
                lambda x=x, fused=fused: core.down_hdiag(x, fused),
            )

            for projection, w, h in (
                ("13", fused, hfused[expert].to(core.DEVICE)),
                ("2", down, hdown),
            ):
                energy = rec.gpu(
                    "dequant_proxy_gemvs", f"expert.weighted_energy.P{projection}",
                    lambda w=w, h=h: core.weighted_energy(w, h),
                )
                frozen_errors: dict[tuple[str, str], float] = {}
                if args.implementation == "exact-gemm":
                    frozen_specs = [
                        (tier, variant)
                        for tier in selected_tiers
                        if tier not in exact_search_tiers
                        for variant in core.variants(tier)
                    ]
                    frozen_errors = rec.gpu(
                        "dequant_proxy_gemvs",
                        f"expert.frozen_dequant_weighted_error_fused.P{projection}",
                        lambda w=w, h=h, projection=projection,
                               frozen_specs=frozen_specs: core.frozen_weighted_errors_batched(
                            w,
                            h,
                            plane_data,
                            cb,
                            expert,
                            projection,
                            frozen_specs,
                        ),
                    )
                for tier in selected_tiers:
                    for variant in core.variants(tier):
                        q: torch.Tensor | None = None
                        err = math.nan
                        if tier in exact_search_tiers and args.implementation == "exact-gemm":
                            q, encode_meta = rec.gpu(
                                "codebook_distance_sweeps",
                                f"expert.{tier}_exact_gemm_encode.P{projection}.{variant}",
                                lambda w=w, projection=projection, tier=tier, variant=variant: core.encode_dequant_row_exact_gemm(
                                    w,
                                    plane_data["d4_k4096"],
                                    cb[(tier, variant, projection)],
                                    expert,
                                    projection,
                                    plan=exact_codebook_plans[(tier, projection)],
                                    audit_assignments=args.audit_codeword_assignments,
                                ),
                            )
                            encoder = (
                                "full-codebook TF32x3 tensor-core top2 + "
                                "bound-gated fused IEEE-FP32 verification"
                            )
                        elif tier == "d4_k2048" and args.implementation == "optimized":
                            _, candidate_slab = resident_candidates[projection]
                            q, encode_meta = rec.gpu(
                                "codebook_distance_sweeps",
                                f"expert.k2048_resident_encode.P{projection}.{variant}",
                                lambda w=w, projection=projection,
                                       candidate_slab=candidate_slab: core.encode_dequant_row_resident(
                                    w,
                                    plane_data["d4_k4096"],
                                    expert,
                                    projection,
                                    plane_data["d4_k4096"][f"codes{projection}"][expert],
                                    candidate_slab,
                                ),
                            )
                            encoder = "resident BF16 top-64 coarse candidate slab; quality-gated"
                        elif tier in exact_search_tiers and args.implementation == "serial":
                            shortlist, first_excluded_sq = certified_shortlists[
                                (tier, projection)
                            ]
                            q, encode_meta = rec.gpu(
                                "codebook_distance_sweeps",
                                f"expert.{tier}_exact_encode.P{projection}.{variant}",
                                lambda w=w, projection=projection, tier=tier, variant=variant,
                                       shortlist=shortlist, first_excluded_sq=first_excluded_sq: core.encode_dequant_row(
                                    w,
                                    plane_data["d4_k4096"],
                                    cb[(tier, variant, projection)],
                                    expert,
                                    projection,
                                    plane_data["d4_k4096"][f"codes{projection}"][expert],
                                    cb[("d4_k4096", "base", projection)],
                                    shortlist,
                                    first_excluded_sq,
                                    audit_assignments=args.audit_codeword_assignments,
                                ),
                            )
                            encoder = (
                                "exact shortlist + triangle certification + exhaustive fallback"
                            )
                        elif args.implementation == "exact-gemm":
                            q = None
                            err = frozen_errors[(tier, variant)]
                            encode_meta = {"certified_vectors": 0, "fallback_vectors": 0}
                            encoder = "fused all-tier frozen-plane BF16 dequant + FP32 weighted-SSE"
                        else:
                            q = rec.gpu(
                                "dequant_proxy_gemvs",
                                f"expert.frozen_dequant.P{projection}.{tier}.{variant}",
                                lambda tier=tier, variant=variant, projection=projection: core.dequant_row(
                                    plane_data[tier], cb[(tier, variant, projection)], expert, projection
                                ),
                            )
                            encode_meta = {"certified_vectors": 0, "fallback_vectors": 0}
                            encoder = "frozen uniform-plane codes/scales"

                        if q is not None:
                            err = rec.gpu(
                                "dequant_proxy_gemvs",
                                f"expert.weighted_error.P{projection}.{tier}.{variant}",
                                lambda w=w, q=q, h=h: core.weighted_error(w, q, h),
                            )
                        if not (math.isfinite(energy) and math.isfinite(err)):
                            raise RuntimeError((layer, expert, projection, tier, variant, energy, err))
                        certified_vectors += int(encode_meta.get("certified_vectors", 0))
                        fallback_vectors += int(encode_meta.get("fallback_vectors", 0))
                        approximate_vectors += int(encode_meta.get("approximate_vectors", 0))
                        exact_vectors += int(encode_meta.get("exact_vectors", 0))
                        scientific_rows.append({
                            "schema": "banana-smasher-solver-cell-tier-v1",
                            "layer": layer,
                            "expert": expert,
                            "projection": projection,
                            "cell": f"L{layer:03d}.E{expert:03d}.P{projection}",
                            "tier": tier,
                            "variant": variant,
                            "weighted_sse": err,
                            "teacher_energy": energy,
                            "relative_weighted_error": err / energy if energy else math.inf,
                            "routed_rows": int(counts[expert]),
                            "n_windows": actual_windows,
                            "encoder": encoder,
                            "codebook_md5": cb_md5[(tier, variant, projection)],
                            **(
                                {
                                    "codeword_assignment_sha256": encode_meta[
                                        "codeword_assignment_sha256"
                                    ],
                                    "codeword_assignment_count": encode_meta[
                                        "codeword_assignment_count"
                                    ],
                                    "codeword_assignment_dtype": encode_meta[
                                        "codeword_assignment_dtype"
                                    ],
                                }
                                if "codeword_assignment_sha256" in encode_meta
                                else {}
                            ),
                        })
                        row_count += 1
                        if q is not None:
                            del q

            del x, hdown
            # Attribute Python loop/dict/control work not covered by explicit phases.
            expert_preclose = time.perf_counter()
            phase_delta = sum(rec.totals[k] - before[k] for k in BUCKETS)
            rec.add_untracked(
                "expert.uninstrumented_python",
                max(0.0, (expert_preclose - expert_t0) - phase_delta),
                expert_t0,
                expert_preclose,
            )
            after = rec.snapshot()
            row = {
                "schema": "solver-profile-expert-row-v1",
                "layer": layer,
                "expert": expert,
                "routed_rows": int(counts[expert]),
                "solver_rows": row_count,
                "elapsed_s": expert_preclose - expert_t0,
                "cumulative_elapsed_s": expert_preclose - mono_started,
                "bucket_seconds": {k: after[k] - before[k] for k in BUCKETS},
                "certified_vectors_cumulative": certified_vectors,
                "fallback_vectors_cumulative": fallback_vectors,
                "approximate_vectors_cumulative": approximate_vectors,
                "exact_vectors_cumulative": exact_vectors,
                "fsync_epoch": time.time(),
            }
            fs0 = time.perf_counter()
            append_fsync(rows_path, row)
            fs1 = time.perf_counter()
            rec.add("python_dispatch", "expert.row_fsync", fs1 - fs0, fs0, fs1)

            if emit_summary and expert % 8 == 7:
                print(
                    f"PROFILE L{layer:03d} E{expert:03d}/255 rows={row_count} "
                    f"elapsed={time.perf_counter() - mono_started:.3f}s",
                    flush=True,
                )
                gc0 = time.perf_counter()
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                gc1 = time.perf_counter()
                rec.add("python_dispatch", "expert.gc_empty_cache", gc1 - gc0, gc0, gc1)

        del fused_batch, down_batch
        if args.implementation == "serial":
            del fused_one, down_one
        if args.implementation == "exact-gemm":
            prefetched = next_prefetched

    save0 = time.perf_counter()
    scientific_path = out / "SCIENTIFIC_ROWS.pt"
    torch.save(scientific_rows, scientific_path)
    assignments: dict[str, dict[str, Any]] = {}
    for scientific_row in scientific_rows:
        cell = scientific_row["cell"]
        if cell not in assignments or scientific_row["relative_weighted_error"] < assignments[cell]["relative_weighted_error"]:
            assignments[cell] = scientific_row
    codeword_audit_complete = bool(assignments) and all(
        "codeword_assignment_sha256" in row for row in assignments.values()
    )
    if codeword_audit_complete:
        assignment_payload = {
            cell: {
                "tier": row["tier"],
                "variant": row["variant"],
                "codeword_assignment_sha256": row[
                    "codeword_assignment_sha256"
                ],
                "codeword_assignment_count": row["codeword_assignment_count"],
                "codeword_assignment_dtype": row["codeword_assignment_dtype"],
            }
            for cell, row in sorted(assignments.items())
        }
        assignment_scope = "full-codeword-assignment-by-cell"
    else:
        assignment_payload = {
            cell: {"tier": row["tier"], "variant": row["variant"]}
            for cell, row in sorted(assignments.items())
        }
        assignment_scope = "cell-tier-variant-only"
    assignment_digest = assignment_sha256(assignment_payload)
    objective = {
        "selected_cells": len(assignments),
        "assignment_sha256": assignment_digest,
        "assignment_scope": assignment_scope,
        "sum_relative_weighted_error": sum(row["relative_weighted_error"] for row in assignments.values()),
        "sum_weighted_sse": sum(row["weighted_sse"] for row in assignments.values()),
    }
    atomic_json(out / "OBJECTIVE.json", {"schema": "banana-smasher-objective-v1", **objective, "assignment": assignment_payload})
    save1 = time.perf_counter()
    rec.add("python_dispatch", "output.scientific_rows_and_objective", save1 - save0, save0, save1)

    end_preclose = time.perf_counter()
    accounted = sum(rec.totals.values())
    outer_so_far = end_preclose - mono_started
    if outer_so_far > accounted:
        rec.add_untracked("outer.uninstrumented_python", outer_so_far - accounted, mono_started, end_preclose)

    epoch_ended = time.time()
    mono_ended = time.perf_counter()
    outer_s = mono_ended - mono_started
    # Final timer bookkeeping can introduce tiny drift; close it into dispatch.
    accounted = sum(rec.totals.values())
    if outer_s > accounted:
        rec.add_untracked("outer.final_timer_gap", outer_s - accounted, end_preclose, mono_ended)
    accounted = sum(rec.totals.values())

    window_rows = []
    for wid in sorted(rec.windows):
        w = rec.windows[wid]
        total = sum(w.values())
        window_rows.append({
            "window_id": wid,
            "start_s": wid * args.sample_seconds,
            "end_s": min(outer_s, (wid + 1) * args.sample_seconds),
            "bucket_seconds": w,
            "accounted_s": total,
        })

    summary = {
        "schema": "banana-smasher-solver-profile-v1",
        "implementation": args.implementation,
        "status": "PASS",
        "layer": layer,
        "tiers": list(selected_tiers),
        "representative_selection": "explicitly selected layer and manifest-bound inputs",
        "scientific_source": "banana_smasher.solver_core",
        "windows": actual_windows,
        "sample_seconds": args.sample_seconds,
        "experts": 256,
        "cells": 512,
        "solver_rows": row_count,
        "expected_solver_rows": 512 * (
            len(selected_tiers) + len(set(selected_tiers) & core.REPAIRED)
        ),
        "certified_vectors": certified_vectors,
        "fallback_vectors": fallback_vectors,
        "approximate_vectors": approximate_vectors,
        "exact_vectors": exact_vectors,
        "expert_batch_size": expert_batch_size,
        "resident_staging_root": (
            str(args.staging_root.resolve()) if args.staging_root is not None else None
        ),
        "weight_pipeline": (
            "bulk resident-host slabs + dedicated-CUDA-stream double buffer"
            if args.implementation == "exact-gemm"
            else "synchronous"
        ),
        "objective": objective,
        "audit_codeword_assignments": args.audit_codeword_assignments,
        "scientific_rows": str(scientific_path),
        "epoch_started": epoch_started,
        "epoch_ended": epoch_ended,
        "outer_wall_s": outer_s,
        "accounted_s": accounted,
        "accounting_error_s": outer_s - accounted,
        "bucket_seconds": rec.totals,
        "async_work_seconds": rec.async_work_totals,
        "async_subphases": rec.async_subphases,
        "host_staging_inclusive_work_s": (
            rec.totals["host_staging"] + rec.async_work_totals["host_staging"]
        ),
        "bucket_fraction": {k: rec.totals[k] / outer_s for k in BUCKETS},
        "subphases": rec.subphases,
        "sample_windows": window_rows,
        "plane_paths": {k: str(v) for k, v in paths.items()},
        "profile_rows": str(rows_path),
        "bucket_definition": {
            "codebook_distance_sweeps": (
                "all-candidate TF32x3 tensor-core top2 plus bound-gated fused IEEE-FP32 verification"
                if args.implementation == "exact-gemm"
                else "certified k2048 shortlist construction and exact nearest-codebook search/fallback"
            ),
            "dequant_proxy_gemvs": "true-weight down-H proxy GEMMs plus frozen-plane dequant and H-weighted energy/error scoring",
            "hessian_proxy_reads": "capture receipt/mmap reads, fused diagonal X^T X proxy construction, expert capture routing reads",
            "host_staging": "exclusive critical-path wall for source/hash/atomic-completion staging, mmap/codebook load, CPU slabs, prefetch submission and uncovered wait; complete overlapped H2D/dequant device work is reported separately in async_work_seconds/async_subphases",
            "python_dispatch": "CUDA-event wall minus device time, Python control/serialization/fsync/GC, and closed residual",
        },
    }
    if summary["solver_rows"] != summary["expected_solver_rows"]:
        raise RuntimeError(summary)
    atomic_json(out / "PROFILE_SUMMARY.json", summary)
    if emit_summary:
        print(json.dumps(summary, sort_keys=True), flush=True)
    return summary


if __name__ == "__main__":
    main()
