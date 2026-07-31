from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .bootstrap import bootstrap_container, container_recipe_path
from .contract import (
    PackValidationError,
    export_pack,
    verify_pack,
    verify_serve_compatibility,
)
from .repack import repack_to_safetensors
from .validation import ValidationError, validate_artifact


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smash",
        description="Five fail-closed bs-pack lifecycle verbs.",
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
    return parser


def _emit(value: dict[str, Any], *, stream: Any | None = None) -> None:
    if stream is None:
        stream = sys.stdout
    stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    tokens = list(sys.argv[1:] if argv is None else argv)
    reported_command = tokens[0] if tokens else None
    if reported_command == "validate-pack":
        # Compatibility spelling for reproducibility automation. Keep the five
        # primary lifecycle verbs and their help surface stable.
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
        else:  # pragma: no cover - argparse guarantees the choices
            parser.error(f"unsupported command {args.command!r}")
            return 2
    except (
        PackValidationError,
        ValidationError,
        FileExistsError,
        OSError,
        ValueError,
    ) as exc:
        _emit(
            {
                "status": "FAIL",
                "command": reported_command or args.command,
                "error": str(exc),
                "error_type": type(exc).__name__,
            },
            stream=sys.stderr,
        )
        return 2
    _emit(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
