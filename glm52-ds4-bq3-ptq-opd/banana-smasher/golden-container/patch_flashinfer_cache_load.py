#!/usr/bin/env python3
"""Allow FlashInfer's sealed compiled cache to satisfy DISABLE_JIT startup.

FlashInfer 0.6.14's ``JitSpec.build_and_load`` unconditionally calls ``build``;
``build`` then raises whenever ``FLASHINFER_DISABLE_JIT`` is set, even when the
exact compiled shared object already exists. This source-hash-gated patch loads
that existing object under the normal FlashInfer lock and retains the upstream
build path whenever JIT is enabled. A missing cached object still fails closed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PREIMAGE = "836f04a87bca0a36e86c04219aac2aabbe28d87c91a577584e81f8c59e4c7a7c"

BLOCK_OLD = '''        with FileLock(self.lock_path, thread_local=False):
            so_path = self.jit_library_path
            verbose = os.environ.get("FLASHINFER_JIT_VERBOSE", "0") == "1"
            self.build(verbose, need_lock=False)
            result = self.load(so_path)
'''

BLOCK_NEW = '''        with FileLock(self.lock_path, thread_local=False):
            so_path = self.jit_library_path
            verbose = os.environ.get("FLASHINFER_JIT_VERBOSE", "0") == "1"
            if os.environ.get("FLASHINFER_DISABLE_JIT"):
                if not so_path.is_file():
                    raise MissingJITCacheError(
                        "JIT compilation is disabled via FLASHINFER_DISABLE_JIT "
                        "and the required cached shared object is absent: "
                        f"{so_path}",
                        spec=self,
                    )
            else:
                self.build(verbose, need_lock=False)
            result = self.load(so_path)
'''


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def patched_bytes(target: Path) -> bytes:
    data = target.read_bytes()
    before = sha256_bytes(data)
    if before != PREIMAGE:
        raise SystemExit(f"flashinfer core.py preimage drift: {before} != {PREIMAGE}")
    text = data.decode()
    if text.count(BLOCK_OLD) != 1:
        raise SystemExit("flashinfer build_and_load patch anchor is not unique")
    text = text.replace(BLOCK_OLD, BLOCK_NEW)
    compile(text, str(target), "exec")
    return text.encode()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()

    before = args.target.read_bytes()
    after = patched_bytes(args.target)
    args.target.write_bytes(after)
    receipt = {
        "schema": "genesis-flashinfer-sealed-cache-load-patch-v1",
        "target": str(args.target),
        "preimage_sha256": sha256_bytes(before),
        "postimage_sha256": sha256_bytes(after),
        "flashinfer_version": "0.6.14",
        "jit_disabled_loads_existing_cache": True,
        "jit_disabled_missing_cache_fails_closed": True,
        "jit_enabled_upstream_build_path_preserved": True,
        "truth_label": "PUBLIC_CANON_IQ3_WIRE; NOT P943 native TRUE-C",
    }
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(receipt["postimage_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
