#!/usr/bin/env python3
"""Exercise every dry-run path while all common network entry points fail closed."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import socket
import sys
import tempfile
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from banana_smasher import cli, core  # noqa: E402


STAGES = (
    "init", "capture", "anchors", "anchor-mix", "grid", "solve", "retrodict",
    "build", "measure", "calibrate", "resolve", "repair", "pack", "serve",
    "eval", "status", "verify",
)


def main() -> int:
    attempts = []

    def blocked(*args, **kwargs):
        attempts.append({"args": repr(args[:2]), "kwargs": sorted(kwargs)})
        raise RuntimeError("network forbidden by offline verifier")

    plans = {}
    with tempfile.TemporaryDirectory() as temporary:
        workspace = Path(temporary) / "workspace"
        with (
            patch.object(socket.socket, "connect", blocked),
            patch.object(socket, "create_connection", blocked),
            patch.object(core, "urlopen", blocked),
        ):
            for stage in STAGES:
                stdout = StringIO()
                stderr = StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    status = cli.main(["--workspace", str(workspace), stage, "--dry-run"])
                if status != 0:
                    raise RuntimeError("{} dry-run failed: {}".format(stage, stderr.getvalue()))
                plan = json.loads(stdout.getvalue())
                if plan.get("stage") != stage or plan.get("mode") != "dry-run" or plan.get("offline") is not True:
                    raise RuntimeError("{} emitted an invalid dry-run contract".format(stage))
                plans[stage] = plan["receipt"]
        if workspace.exists():
            raise RuntimeError("dry-run created workspace output")
    if attempts:
        raise RuntimeError("dry-run attempted network access: " + json.dumps(attempts, sort_keys=True))
    print(json.dumps({
        "schema": "banana-smasher-offline-dry-run-verification-v1",
        "status": "PASS",
        "network_blocked": True,
        "network_attempts": 0,
        "workspace_created": False,
        "stages": list(STAGES),
        "receipts": plans,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
