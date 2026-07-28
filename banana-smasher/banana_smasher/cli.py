"""Command-line surface for Banana Smasher."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from .core import run_generic_stage, run_init, stage_plan, status, verify_manifest, verify_self_contained


STAGES = (
    "init", "capture", "anchors", "anchor-mix", "grid", "solve", "retrodict",
    "build", "measure", "calibrate", "resolve", "repair", "pack", "serve",
    "eval", "status",
)


def _dry_run(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dry-run", action="store_true", help="print the offline execution plan and write nothing")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="smash",
        description="Self-contained model profile to anchors, solve, wire, repair, serve, and eval prototype.",
    )
    root.add_argument(
        "--workspace",
        default=str(Path.cwd() / "workspace"),
        help="stage workspace root (default: ./workspace)",
    )
    commands = root.add_subparsers(dest="command", required=True)

    init = commands.add_parser("init", help="profile a local model config or Hugging Face model")
    init.add_argument("--model")
    init.add_argument("--budget-bytes", type=int)
    init.add_argument("--node-ram", type=float, help="node RAM in decimal GB")
    _dry_run(init)

    capture = commands.add_parser("capture", help="plan teacher-forward capture and KLD banks")
    capture.add_argument("--windows", default="balanced64")
    _dry_run(capture)

    anchors = commands.add_parser("anchors", help="build and measure uniform vertical anchors")
    anchors.add_argument("--tiers", default="qtip3,qtip2,d4_k1024,d4_k2048,d4_k4096,mxfp4")
    _dry_run(anchors)

    anchor_mix = commands.add_parser("anchor-mix", help="measure mixed-tier additivity interaction")
    anchor_mix.add_argument("--pattern", default="auto")
    _dry_run(anchor_mix)

    grid = commands.add_parser("grid", help="assemble the SHA-sealed priced grid")
    _dry_run(grid)

    solve = commands.add_parser("solve", help="paired-swap seed followed by SCIP solve")
    solve.add_argument("--weights", default="uniform")
    solve.add_argument("--scip-seconds", type=int, default=7200)
    _dry_run(solve)

    retrodict = commands.add_parser("retrodict", help="refuse if measured-wire prediction error exceeds the gate")
    retrodict.add_argument("--threshold-percent", type=float, default=5.0)
    _dry_run(retrodict)

    build = commands.add_parser("build", help="materialize the assigned wire with per-cell seals")
    build.add_argument("--shards", type=int, default=1)
    build.add_argument("--peer-stream", action="store_true")
    _dry_run(build)

    measure = commands.add_parser("measure", help="read the KLD rail for the sealed wire")
    measure_mode = measure.add_mutually_exclusive_group()
    measure_mode.add_argument("--balanced64", action="store_true", default=True)
    measure_mode.add_argument("--full", action="store_true")
    _dry_run(measure)

    calibrate = commands.add_parser("calibrate", help="fit residual families from all measured wires")
    calibrate.add_argument("--ridge-lambda", type=float, default=1e-4)
    _dry_run(calibrate)

    resolve = commands.add_parser("resolve", help="solve definitively on the corrected grid")
    resolve.add_argument("--scip-seconds", type=int, default=7200)
    _dry_run(resolve)

    repair = commands.add_parser("repair", help="dose repair from the sealed-wire inventory")
    repair.add_argument("--updates", type=int, default=24)
    _dry_run(repair)

    pack = commands.add_parser("pack", help="build a serving pack from the sealed wire")
    pack.add_argument("--format", default="mixed-tier-pack-v1")
    _dry_run(pack)

    serve = commands.add_parser("serve", help="serve the pack behind an OpenAI-compatible endpoint")
    serve.add_argument("--docker", action="store_true")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    _dry_run(serve)

    evaluate = commands.add_parser("eval", help="run greedy or sampled EvalPlus through the endpoint")
    evaluate.add_argument("--suite", default="humaneval+")
    evaluate.add_argument("--samples", type=int, default=1)
    evaluate.add_argument("--endpoint", default="http://127.0.0.1:8000/v1")
    _dry_run(evaluate)

    ledger = commands.add_parser("status", help="seal and print the stage receipt ledger")
    _dry_run(ledger)

    verify = commands.add_parser("verify", help="verify manifest closure and self-containment")
    verify.add_argument("--manifest", action="store_true")
    verify.add_argument("--self-contained", action="store_true")
    _dry_run(verify)
    return root


def dispatch(namespace: Any) -> Any:
    if namespace.command == "init":
        return run_init(namespace)
    if namespace.command == "status":
        return status(namespace)
    if namespace.command == "verify":
        if namespace.dry_run:
            return stage_plan("verify", namespace)
        checks = []
        if namespace.manifest or not namespace.self_contained:
            checks.append(verify_manifest())
        if namespace.self_contained or not namespace.manifest:
            checks.append(verify_self_contained())
        failed = [check for check in checks if check["status"] != "PASS"]
        result = {
            "schema": "banana-smasher-verification-v1",
            "status": "PASS" if not failed else "FAIL",
            "checks": checks,
        }
        if failed:
            raise VerificationError(result)
        return result
    return run_generic_stage(namespace.command, namespace)


class VerificationError(RuntimeError):
    def __init__(self, result: Any):
        super().__init__(json.dumps(result, indent=2, sort_keys=True))
        self.result = result


def main(argv: Any = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        result = dispatch(arguments)
    except VerificationError as exc:
        print(json.dumps(exc.result, indent=2, sort_keys=True))
        return 1
    except Exception as exc:
        print("smash: {}: {}".format(type(exc).__name__, exc), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
