#!/usr/bin/env python3
"""Container entrypoint for a mounted GENESIS export pack."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from typing import Any
import urllib.request

from pack_contract import SCHEMA_VERSION, PackValidationError, validate_pack
from perf_gate import evaluate as evaluate_perf


def pack_hash_workers() -> int:
    workers = int(os.environ.get("GENESIS_PACK_HASH_WORKERS", "32"))
    if not 1 <= workers <= 64:
        raise PackValidationError(
            f"GENESIS_PACK_HASH_WORKERS must be in [1,64], got {workers}"
        )
    return workers


def prepare_pack(source: str, download_root: str | Path) -> Path:
    if not source.startswith(("http://", "https://")):
        root = Path(source).resolve()
        if not (root / "MANIFEST.json").is_file():
            raise PackValidationError(f"pack MANIFEST.json not found under {root}")
        return root

    destination = Path(download_root).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "export-pack.tar"
    with urllib.request.urlopen(source, timeout=3600) as response, archive.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=8 << 20)
    extracted = destination / "pack"
    extracted.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, mode="r:*") as bundle:
        for member in bundle.getmembers():
            target = (extracted / member.name).resolve()
            if extracted not in target.parents and target != extracted:
                raise PackValidationError(f"archive path escapes pack root: {member.name!r}")
            if member.issym() or member.islnk():
                raise PackValidationError(f"archive links are forbidden: {member.name!r}")
        bundle.extractall(extracted)
    candidates = [extracted] + sorted(path.parent for path in extracted.rglob("MANIFEST.json"))
    roots = [path for path in candidates if (path / "MANIFEST.json").is_file()]
    if len(set(roots)) != 1:
        raise PackValidationError(
            f"downloaded archive must contain exactly one MANIFEST.json; found {len(set(roots))}"
        )
    return roots[0].resolve()


def verify(pack_source: str | Path) -> dict[str, object]:
    source = str(pack_source)
    if source.startswith(("http://", "https://")):
        raise PackValidationError(
            "URL packs are not accepted by verify; download explicitly or use serve"
        )
    pack = validate_pack(
        Path(source),
        expected_schema_version=SCHEMA_VERSION,
        workers=pack_hash_workers(),
    )
    result: dict[str, object] = {
        "status": "PASS",
        "schema": "genesis-container-verification-v1",
        "container_schema_version": SCHEMA_VERSION,
        "pack": pack,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def build_server_command(
    pack_root: str | Path,
    validation: dict[str, Any],
    mission: str | Path,
    *,
    port: int,
    runtime_root: str | Path = "/opt/genesis/runtime",
) -> tuple[list[str], dict[str, str]]:
    pack = Path(pack_root).resolve()
    run_root = Path(mission).resolve()
    serving = validation["serving"]
    envelope = validation["resident_envelope"]
    command = [
        sys.executable,
        str(Path(runtime_root) / "mixed_prefill_server.py"),
        "--mission", str(run_root),
        "--source-host", "local",
        "--source-root", str(pack / str(envelope["root"])),
        "--artifact", str(pack / str(serving["artifact"])),
        "--tokenizer-json", str(pack / str(serving["tokenizer"])),
        "--port", str(port),
    ]
    environment = {
        "GENESIS_PRODUCT_BYTES": str(envelope["bytes"]),
        "GENESIS_PRODUCT_FILES": str(envelope["files"]),
        "GENESIS_PRODUCT_INVENTORY_SHA256": str(
            envelope["sealed_source_inventory_sha256"]
        ),
        "GENESIS_MODEL_ID": str(validation["model_id"]),
        "GENESIS_MANIFEST_PATH": str(pack / "MANIFEST.json"),
        "P530_FILE_BACKED_ENVELOPE": "1",
        "P530_PREFILL_MODE": os.environ.get("P530_PREFILL_MODE", "dense_all"),
        "P525_DENSE_THRESHOLD": os.environ.get("P525_DENSE_THRESHOLD", "64"),
        "P525_DENSE_CHUNK_ROWS": os.environ.get("P525_DENSE_CHUNK_ROWS", "1024"),
        "TRITON_CACHE_DIR": os.environ.get(
            "TRITON_CACHE_DIR", "/opt/genesis/triton-cache"
        ),
    }
    return command, environment


def build_startup_receipt(
    *,
    started_at: float,
    models_ready_at: float,
    request_started_at: float,
    request_finished_at: float,
    response: dict[str, Any],
) -> dict[str, Any]:
    metrics = response.get("mixed_tier", {})
    ttft = float(metrics["ttft_seconds"])
    return {
        "status": "PASS",
        "schema": "genesis-container-startup-smoke-v1",
        "id": response.get("id"),
        "model": response.get("model"),
        "bind_seconds": models_ready_at - started_at,
        "first_token_seconds_from_container_start": (
            request_started_at - started_at + ttft
        ),
        "smoke_response_seconds_from_container_start": request_finished_at - started_at,
        "smoke_request_seconds": request_finished_at - request_started_at,
        "ttft_seconds_server": ttft,
        "prefill_tok_s": metrics.get("prefill_tok_s_server"),
        "decode_tok_s": metrics.get("decode_tok_s"),
        "resident_product_bytes": metrics.get("resident_product_bytes"),
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _http_json(url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
    )
    with urllib.request.urlopen(request, timeout=3600) as response:
        return json.load(response)


def serve(pack_source: str) -> int:
    started_at = time.monotonic()
    mission = Path(os.environ.get("GENESIS_MISSION", "/run/genesis")).resolve()
    receipts = mission / "receipts"
    pack = prepare_pack(pack_source, mission / "downloads")
    validation = validate_pack(
        pack,
        expected_schema_version=SCHEMA_VERSION,
        workers=pack_hash_workers(),
    )
    _write_json(receipts / "PACK_VALIDATION.json", validation)
    cache_environment = dict(os.environ)
    cache_environment["GENESIS_RUNTIME"] = "/opt/genesis/runtime"
    cache_environment["GENESIS_WARMUP_ARTIFACT"] = str(
        pack / str(validation["serving"]["artifact"])
    )
    subprocess.run(
        [sys.executable, "/opt/genesis/runtime/warmup_kernels.py"],
        env=cache_environment,
        check=True,
    )
    subprocess.run(
        [sys.executable, "/opt/genesis/runtime/verify_kernel_cache.py"],
        env=cache_environment,
        check=True,
    )
    cache_verified = True
    port = int(os.environ.get("GENESIS_PORT", "8000"))
    command, overrides = build_server_command(pack, validation, mission, port=port)
    environment = dict(os.environ)
    environment.update(overrides)
    print(json.dumps({
        "event": "pack_validated",
        "container_schema_version": SCHEMA_VERSION,
        "model": validation["model_id"],
        "resident_product_bytes": validation["resident_envelope"]["bytes"],
        "server_command": command,
    }, sort_keys=True), flush=True)
    process = subprocess.Popen(command, env=environment)

    def forward(signum: int, _frame: object) -> None:
        if process.poll() is None:
            process.send_signal(signum)

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    try:
        deadline = started_at + float(os.environ.get("GENESIS_START_TIMEOUT", "60"))
        models_url = f"http://127.0.0.1:{port}/v1/models"
        while True:
            if process.poll() is not None:
                raise RuntimeError(f"server exited before health check rc={process.returncode}")
            try:
                models = _http_json(models_url)
                if models.get("data"):
                    break
            except Exception:
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError("/v1/models was not ready within GENESIS_START_TIMEOUT")
            time.sleep(0.25)
        models_ready_at = time.monotonic()
        request_started_at = time.monotonic()
        response = _http_json(
            f"http://127.0.0.1:{port}/v1/completions",
            {
                "model": validation["model_id"],
                "prompt": " warmup" * 2048,
                "max_tokens": 16,
                "temperature": 0,
                "stream": False,
            },
        )
        request_finished_at = time.monotonic()
        smoke = build_startup_receipt(
            started_at=started_at,
            models_ready_at=models_ready_at,
            request_started_at=request_started_at,
            request_finished_at=request_finished_at,
            response=response,
        )
        mixed = response.get("mixed_tier", {})
        usage = response.get("usage", {})
        readiness = evaluate_perf(
            {
                "validity": "fresh-measurement",
                "prompt_tokens": usage.get("prompt_tokens", mixed.get("prompt_tokens", 0)),
                "prefill_tok_s": mixed.get("prefill_tok_s_server"),
                "decode_tok_s": mixed.get("decode_tok_s"),
                "ttft_seconds": mixed.get("ttft_seconds"),
                "decode_kernel_classes": [
                    "_qtip_gemv",
                    "_truevq_d4_gemv",
                    "_truevq_d8_gemv",
                    "_native_mxfp4_gemv",
                ],
                "cache_verified": cache_verified,
                "resident_envelope_verified": (
                    mixed.get("resident_product_bytes")
                    == validation["resident_envelope"]["bytes"]
                ),
            },
            json.loads(Path("/opt/genesis/config/EXPECTED_PERF.json").read_text()),
        )
        smoke["readiness"] = readiness
        smoke["status"] = readiness["status"]
        _write_json(receipts / "STARTUP_SMOKE.json", smoke)
        print("GENESIS_STARTUP_SMOKE " + json.dumps(smoke, sort_keys=True), flush=True)
        if readiness["status"] != "READY":
            process.terminate()
            process.wait(timeout=30)
            return 3
        return process.wait()
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise


def main(argv: list[str]) -> int:
    # Public contract: one positional pack path/URL starts serving. Explicit
    # `serve` and `verify` subcommands remain available for operators/tests.
    if len(argv) == 1:
        return serve("/model")
    command = argv[1]
    if command == "verify":
        source = argv[2] if len(argv) > 2 else "/model"
        verify(source)
        return 0
    if command == "serve":
        source = argv[2] if len(argv) > 2 else "/model"
        return serve(source)
    if len(argv) == 2:
        return serve(command)
    raise SystemExit(f"unknown command: {command}")


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv))
    except PackValidationError as exc:
        print(
            json.dumps({
                "status": "FAIL",
                "container_schema_version": SCHEMA_VERSION,
                "error": str(exc),
            }, sort_keys=True),
            file=sys.stderr,
        )
        raise SystemExit(2)
