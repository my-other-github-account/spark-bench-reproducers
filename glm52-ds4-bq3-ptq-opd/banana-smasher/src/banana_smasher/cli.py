from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import traceback
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .bootstrap import bootstrap_container, container_recipe_path
from .contract import (
    export_pack,
    verify_pack,
    verify_serve_compatibility,
)
from .repack import repack_to_safetensors
from .validation import validate_artifact


def _parse_source_windows(value: str) -> tuple[int, ...]:
    try:
        windows = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "source windows must be a comma-separated list of non-negative integers"
        ) from exc
    if not windows or any(window < 0 for window in windows):
        raise argparse.ArgumentTypeError(
            "source windows must be a non-empty comma-separated list of non-negative integers"
        )
    if len(set(windows)) != len(windows):
        raise argparse.ArgumentTypeError("source windows must be unique")
    return windows


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smash",
        description="Fail-closed bs-pack lifecycle and physical-update commands.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    export = subparsers.add_parser("export", help="export quantizer output to bs-pack")
    export.add_argument("--source-root", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--model-id", required=True)
    export.add_argument("--instance-id", required=True)
    export.add_argument(
        "--link-mode", choices=("hardlink", "copy", "auto"), default="hardlink"
    )
    export.add_argument(
        "--safetensors",
        action="store_true",
        help="also repack all planes into bs-pack.safetensors and verify payload identity",
    )
    export.add_argument(
        "--drop-planes",
        action="store_true",
        help="drop .npy planes only after a verified safetensors repack",
    )

    verify = subparsers.add_parser("verify", help="verify manifest, schema, and bytes")
    verify.add_argument("pack", type=Path)

    serve = subparsers.add_parser(
        "serve-check", help="verify pack/kernel-cache compatibility before vllm serve"
    )
    serve.add_argument("pack", type=Path)
    serve.add_argument("--kernel-cache", type=Path, required=True)
    serve.add_argument("--architecture", required=True)

    validate = subparsers.add_parser(
        "validate", help="run a banked-teacher KLD validation ceremony"
    )
    validate.add_argument("artifact", type=Path)
    validate.add_argument("--bank", required=True)
    validate.add_argument("--check-exposure", action="store_true")
    validate.add_argument("--receipt", type=Path)
    validate.add_argument("--bank-teacher-logits", type=Path)

    bootstrap = subparsers.add_parser(
        "bootstrap", help="build or pull the stock-semantics vLLM container"
    )
    bootstrap.add_argument("--recipe", type=Path)
    bootstrap.add_argument("--context", type=Path, default=Path.cwd())
    bootstrap.add_argument(
        "--image", default="banana_smasher-serve:banana-smasher-candidate"
    )
    bootstrap.add_argument("--docker-bin", default="docker")
    bootstrap.add_argument("--pull", action="store_true")
    bootstrap.add_argument("--receipt", type=Path, default=Path("BOOTSTRAP_RECEIPT.json"))

    solve = subparsers.add_parser(
        "solve", help="solve declared cells or fresh-model VQ tiers with exact search"
    )
    solve.add_argument("--source-root", type=Path, required=True)
    solve.add_argument("--output", type=Path)
    solve.add_argument(
        "--root",
        type=Path,
        help="workflow run root; selects fresh-model multi-layer solve mode",
    )
    solve.add_argument("--layer", type=int)
    solve.add_argument("--layers", default="0-42")
    solve.add_argument("--tiers", default="d4_k2048,d4_k4096")
    solve.add_argument(
        "--tier",
        choices=("qtip3", "qtip2"),
        help="named exact QTIP tier for the public all-cells path",
    )
    solve.add_argument(
        "--all-cells",
        action="store_true",
        help="solve every ordered expert/projection cell for each selected layer",
    )
    solve.add_argument("--windows", type=int, choices=(32, 64), default=32)
    solve.add_argument("--staging-root", type=Path)
    solve.add_argument(
        "--prices-root",
        type=Path,
        help="sealed SOLVER_PRICING_V2 prices/ root to adopt without reprofiling",
    )
    solve.add_argument("--hessian-manifest", type=Path)
    solve.add_argument("--detach", action="store_true")
    solve.add_argument("--device", default="cuda")
    solve.add_argument("--reference-search", action="store_true", help=argparse.SUPPRESS)
    solve.add_argument(
        "--audit-codeword-assignments",
        action="store_true",
        help="hash every exact codeword winner (intended for one-layer parity audits)",
    )
    solve.add_argument("--verbose-receipts", action="store_true", help=argparse.SUPPRESS)
    solve.add_argument(
        "--qtip-profile-config",
        type=Path,
        help="sealed local-input config for a fresh exact QTIP solve",
    )
    solve.add_argument(
        "--qtip-units",
        type=int,
        help="limit an ordered resident QTIP config-directory solve",
    )
    solve.add_argument(
        "--profile-qtip",
        action="store_true",
        help="profile the exact QTIP solve through this public solve verb",
    )
    solve.set_defaults(backend="exact-gemm")

    hessian = subparsers.add_parser(
        "hessian", help="prefetch and seal task-local Hessian/capture banks"
    )
    hessian.add_argument("--run-root", type=Path, required=True)
    hessian.add_argument("--layers", default="0-42")
    hessian.add_argument("--windows", type=int, choices=(32, 64), default=32)
    hessian.add_argument("--detach", action="store_true")

    capture = subparsers.add_parser(
        "capture", help="generate resumable task-local TRAIN capture members"
    )
    capture.add_argument("--run-root", type=Path, required=True)
    capture.add_argument("--model-root", type=Path, required=True)
    capture.add_argument("--meta-root", type=Path, required=True)
    capture.add_argument("--corpus", type=Path, required=True)
    capture.add_argument("--builder", type=Path, required=True)
    capture.add_argument("--layers", default="0-42")
    capture.add_argument("--windows", type=int, choices=(32, 64), default=32)
    capture.add_argument("--microbatch", type=int, default=4)
    capture.add_argument("--detach", action="store_true")

    update = subparsers.add_parser(
        "update",
        help="run one resumable accelerated physical update",
    )
    update.add_argument("--runtime-root", type=Path)
    update.add_argument("--model-root", type=Path)
    update.add_argument("--aot", type=Path)
    update.add_argument("--output", type=Path, required=True)
    update.add_argument("--receipt", type=Path)
    update.add_argument(
        "--segments",
        "--accumulation-segments",
        dest="segments",
        type=int,
        default=8,
    )
    update.add_argument("--window", type=int, default=27)
    update.add_argument(
        "--source-windows",
        type=_parse_source_windows,
        help=(
            "ordered comma-separated corpus/teacher windows composing the logical extent; "
            "defaults to --window only"
        ),
    )
    update.add_argument("--tokens", "--tokens-per-segment", dest="tokens", type=int, default=1024)
    update.add_argument("--layers", type=int, choices=(1, 43), default=43)
    update.add_argument("--learning-rate", type=float, default=1e-4)
    update.add_argument(
        "--backend",
        choices=("accelerated", "reference"),
        default="accelerated",
        help=argparse.SUPPRESS,
    )
    update.add_argument("--resume", action="store_true", default=True)
    update.add_argument("--restart", action="store_true")
    update.add_argument("--verbose-receipts", action="store_true")

    anchor = subparsers.add_parser(
        "anchor", help="seal per-tier calibration/rebalance surfaces from a solve manifest"
    )
    anchor.add_argument("--run-root", type=Path, required=True)
    anchor.add_argument("--detach", action="store_true")

    status = subparsers.add_parser(
        "status", help="inspect manifest stages and detached process identities"
    )
    status.add_argument("--run-root", type=Path, required=True)

    # Keep the long-standing public release verbs last. Some downstream wrappers
    # inspect argparse ordering as part of the compatibility contract.
    bank = subparsers.add_parser(
        "bank", help="build or resume a complete manifest-bound teacher bank"
    )
    bank.add_argument("--model-root", type=Path, required=True)
    bank.add_argument("--corpus", type=Path, required=True)
    bank.add_argument("--windows-manifest", type=Path, required=True)
    bank.add_argument("--output", type=Path, required=True)
    bank.add_argument("--instrument-profile", type=Path)

    evaluate = subparsers.add_parser(
        "evaluate", help="run paired candidate/reference real-axis evaluation"
    )
    evaluate.add_argument("--model-root", type=Path, required=True)
    evaluate.add_argument("--candidate", type=Path, required=True)
    evaluate.add_argument("--reference", type=Path, required=True)
    evaluate.add_argument("--bank", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--resume-from-layer", type=int)
    evaluate.add_argument("--verbose-receipts", action="store_true")
    return parser


def _emit(value: dict[str, Any], *, stream: Any | None = None) -> None:
    if stream is None:
        stream = sys.stdout
    stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _seal_update_failure_receipt(
    receipt: Path,
    exc: BaseException,
    *,
    status: str = "FAIL_EXCEPTION",
    output: Path | None = None,
) -> Path:
    failure = receipt.with_name(f"{receipt.stem}.failure.json")
    failure.parent.mkdir(parents=True, exist_ok=True)
    value = {
        "schema": "banana-smasher-update-failure-v1",
        "status": status,
        "created_unix": time.time(),
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }
    if output is not None:
        checkpoint = Path(f"{output.resolve()}.checkpoint")
        value["resume_location"] = str(checkpoint)
        manifest = checkpoint / "manifest.json"
        if manifest.is_file():
            progress = json.loads(manifest.read_text())
            value["last_committed_segment"] = int(
                progress.get("next_segment_index", 0)
            ) - 1
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    temporary = failure.with_name(f".{failure.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, failure)
    directory_fd = os.open(failure.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return failure


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    tokens = list(sys.argv[1:] if argv is None else argv)
    reported_command = tokens[0] if tokens else None
    if reported_command == "validate-pack":
        # Compatibility spelling for reproducibility automation.
        tokens[0] = "verify"
    args = parser.parse_args(tokens)
    try:
        if args.command == "export":
            if args.drop_planes and not args.safetensors:
                raise ValueError("--drop-planes requires --safetensors")
            manifest = export_pack(
                source_root=args.source_root,
                output=args.output,
                model_id=args.model_id,
                instance_id=args.instance_id,
                link_mode=args.link_mode,
            )
            receipt = verify_pack(args.output)
            result = {
                **receipt,
                "command": "export",
                "output": str(args.output.resolve()),
                "file_count": len(manifest["files"]),
            }
            if args.safetensors:
                result["repack"] = repack_to_safetensors(
                    args.output,
                    drop_planes=args.drop_planes,
                )
        elif args.command == "verify":
            result = {
                **verify_pack(args.pack),
                "command": reported_command or "verify",
            }
        elif args.command == "serve-check":
            result = {
                **verify_serve_compatibility(
                    args.pack,
                    args.kernel_cache,
                    architecture=args.architecture,
                ),
                "command": "serve-check",
            }
        elif args.command == "validate":
            result = {
                **validate_artifact(
                    args.artifact,
                    bank=args.bank,
                    check_exposure=args.check_exposure,
                    receipt_path=args.receipt,
                    bank_teacher_logits=args.bank_teacher_logits,
                ),
                "command": "validate",
            }
        elif args.command == "bootstrap":
            result = {
                **bootstrap_container(
                    recipe=args.recipe or container_recipe_path(),
                    context=args.context,
                    image=args.image,
                    docker_bin=args.docker_bin,
                    receipt_path=args.receipt,
                    pull=args.pull,
                ),
                "command": "bootstrap",
            }
        elif args.command == "solve":
            if args.qtip_profile_config is not None:
                if args.root is None or args.layer is None:
                    raise ValueError("--qtip-profile-config requires --root and --layer")
                from .solver_qtip_profile import main as qtip_profile_main
                from .solver_qtip_profile import main_many as qtip_profile_main_many

                if args.qtip_profile_config.is_dir():
                    qtip_profile_main_many(
                        args.qtip_profile_config,
                        args.root,
                        args.layer,
                        limit=args.qtip_units,
                        profile_mode=args.profile_qtip,
                    )
                else:
                    if args.qtip_units not in (None, 1):
                        raise ValueError(
                            "--qtip-units above 1 requires a config directory"
                        )
                    qtip_profile_main(
                        args.qtip_profile_config,
                        args.root,
                        args.layer,
                        profile_mode=args.profile_qtip,
                    )
                return 0
            if args.root is not None:
                if args.output is not None:
                    raise ValueError("fresh-model workflow mode refuses --output; use --root")
                from .workflow import (
                    launch_detached,
                    parse_csv,
                    parse_layers,
                    run_fresh_solve,
                )

                if args.detach:
                    detached_tokens = [token for token in tokens if token != "--detach"]
                    result = launch_detached(
                        run_root=args.root,
                        verb="solve",
                        argv=detached_tokens,
                    )
                else:
                    result = run_fresh_solve(
                        run_root=args.root,
                        source_root=args.source_root,
                        layers=(
                            [args.layer]
                            if args.layer is not None
                            else parse_layers(args.layers)
                        ),
                        tiers=parse_csv(args.tiers),
                        windows=args.windows,
                        staging_root=args.staging_root,
                        reference_search=args.reference_search,
                        hessian_manifest=args.hessian_manifest,
                        prices_root=args.prices_root,
                        audit_codeword_assignments=args.audit_codeword_assignments,
                    )
            else:
                if args.output is None:
                    raise ValueError("solve requires --output or fresh-model --root")
                if args.detach:
                    raise ValueError("--detach requires fresh-model --root")
                # Torch/Triton stay lazy so pack-only commands keep the light install.
                from .solve import run_solve

                result = run_solve(
                    source_root=args.source_root,
                    output=args.output,
                    device=args.device,
                    reference_search=args.reference_search,
                    verbose_receipts=args.verbose_receipts,
                )
        elif args.command == "hessian":
            from .workflow import launch_detached, parse_layers, run_hessian

            if args.detach:
                detached_tokens = [token for token in tokens if token != "--detach"]
                result = launch_detached(
                    run_root=args.run_root,
                    verb="hessian",
                    argv=detached_tokens,
                )
            else:
                result = run_hessian(
                    run_root=args.run_root,
                    layers=parse_layers(args.layers),
                    windows=args.windows,
                )
        elif args.command == "capture":
            from .capture_source import run_capture
            from .workflow import launch_detached, parse_layers

            if args.detach:
                detached_tokens = [token for token in tokens if token != "--detach"]
                result = launch_detached(
                    run_root=args.run_root,
                    verb="capture",
                    argv=detached_tokens,
                )
            else:
                result = run_capture(
                    run_root=args.run_root,
                    model_root=args.model_root,
                    meta_root=args.meta_root,
                    corpus=args.corpus,
                    builder=args.builder,
                    layers=parse_layers(args.layers),
                    windows=args.windows,
                    microbatch=args.microbatch,
                )
        elif args.command == "update":
            # Torch is intentionally lazy: pack-only lifecycle commands remain
            # usable in the lightweight release environment.
            if args.runtime_root is None or args.model_root is None or args.aot is None:
                raise ValueError("--runtime-root, --model-root, and --aot are required")
            from .update import run_minimal_update

            previous_sigterm = signal.getsignal(signal.SIGTERM)

            def interrupt_for_signal(signum, _frame):
                raise KeyboardInterrupt(f"received signal {signum}")

            signal.signal(signal.SIGTERM, interrupt_for_signal)
            try:
                result = run_minimal_update(
                    runtime_root=args.runtime_root,
                    model_root=args.model_root,
                    aot=args.aot,
                    receipt=args.receipt
                    or args.output.with_name(f"{args.output.name}.receipt.json"),
                    output=args.output,
                    window=args.window,
                    source_windows=args.source_windows,
                    tokens=args.tokens,
                    learning_rate=args.learning_rate,
                    layers=args.layers,
                    accumulation_segments=args.segments,
                    backend=args.backend,
                    resume=args.resume,
                    restart=args.restart,
                    verbose_receipts=args.verbose_receipts,
                )
            finally:
                signal.signal(signal.SIGTERM, previous_sigterm)
            result = {**result, "command": "update"}
            if not str(result.get("status", "")).startswith("PASS"):
                _emit(result, stream=sys.stderr)
                return 2
        elif args.command == "bank":
            from .bank import build_bank

            result = build_bank(
                model_root=args.model_root,
                corpus=args.corpus,
                windows_manifest=args.windows_manifest,
                output=args.output,
                instrument_profile=args.instrument_profile,
            )
        elif args.command == "evaluate":
            from .evaluate import evaluate_paired

            result = evaluate_paired(
                model_root=args.model_root,
                candidate=args.candidate,
                reference=args.reference,
                bank=args.bank,
                output=args.output,
                resume_from_layer=args.resume_from_layer,
                verbose_receipts=args.verbose_receipts,
            )
        elif args.command == "anchor":
            from .workflow import launch_detached, run_anchor

            if args.detach:
                detached_tokens = [token for token in tokens if token != "--detach"]
                result = launch_detached(
                    run_root=args.run_root,
                    verb="anchor",
                    argv=detached_tokens,
                )
            else:
                result = run_anchor(run_root=args.run_root)
        elif args.command == "status":
            from .workflow import workflow_status

            result = workflow_status(run_root=args.run_root)
        else:  # pragma: no cover - argparse guarantees the choices
            parser.error(f"unsupported command {args.command!r}")
            return 2
    except KeyboardInterrupt as exc:
        if args.command != "update":
            raise
        receipt_path = args.receipt or args.output.with_name(
            f"{args.output.name}.receipt.json"
        )
        failure_receipt = _seal_update_failure_receipt(
            receipt_path,
            exc,
            status="INTERRUPTED_RESUMABLE",
            output=args.output,
        )
        _emit(
            {
                "status": "INTERRUPTED_RESUMABLE",
                "command": "update",
                "error": str(exc),
                "error_type": type(exc).__name__,
                "failure_receipt": str(failure_receipt),
                "resume_location": str(Path(f"{args.output.resolve()}.checkpoint")),
            },
            stream=sys.stderr,
        )
        return 130
    except Exception as exc:
        failure_receipt = None
        if args.command == "update":
            receipt_path = args.receipt or args.output.with_name(
                f"{args.output.name}.receipt.json"
            )
            failure_receipt = _seal_update_failure_receipt(
                receipt_path, exc, output=args.output
            )
        _emit(
            {
                "status": "FAIL",
                "command": reported_command or args.command,
                "error": str(exc),
                "error_type": type(exc).__name__,
                "failure_receipt": (
                    str(failure_receipt) if failure_receipt is not None else None
                ),
            },
            stream=sys.stderr,
        )
        return 2
    _emit(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
