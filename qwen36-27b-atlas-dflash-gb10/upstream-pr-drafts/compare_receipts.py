#!/usr/bin/env python3
"""Compare AR and DFlash/OpenAI-compatible JSON receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from typing import Any


def _load(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _content(receipt: dict[str, Any]) -> str:
    if isinstance(receipt.get("content"), str):
        return receipt["content"]
    if isinstance(receipt.get("output"), str):
        return receipt["output"]
    if isinstance(receipt.get("text"), str):
        return receipt["text"]
    return json.dumps(receipt.get("choices", receipt), sort_keys=True)


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _usage(receipt: dict[str, Any]) -> dict[str, Any] | None:
    usage = receipt.get("usage")
    if isinstance(usage, dict):
        return usage
    return None


def _token_ids(receipt: dict[str, Any]) -> list[int] | None:
    ids = receipt.get("token_ids")
    if isinstance(ids, list) and all(isinstance(v, int) for v in ids):
        return ids
    return None


def compare(ar_path: str, dflash_path: str) -> dict[str, Any]:
    ar = _load(ar_path)
    dflash = _load(dflash_path)
    ar_text = _content(ar)
    dflash_text = _content(dflash)
    ar_tokens = _token_ids(ar)
    dflash_tokens = _token_ids(dflash)
    token_ids_equal = None
    if ar_tokens is not None and dflash_tokens is not None:
        token_ids_equal = ar_tokens == dflash_tokens
    result = {
        "ar_receipt": ar_path,
        "dflash_receipt": dflash_path,
        "content_equal": ar_text == dflash_text,
        "content_hash_equal": _hash_text(ar_text) == _hash_text(dflash_text),
        "ar_content_sha256": _hash_text(ar_text),
        "dflash_content_sha256": _hash_text(dflash_text),
        "usage_equal": _usage(ar) == _usage(dflash),
        "ar_usage": _usage(ar),
        "dflash_usage": _usage(dflash),
        "token_ids_equal": token_ids_equal,
        "ar_token_count": len(ar_tokens) if ar_tokens is not None else None,
        "dflash_token_count": len(dflash_tokens) if dflash_tokens is not None else None,
    }
    result["match"] = (
        result["content_equal"]
        and result["usage_equal"]
        and (token_ids_equal is not False)
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ar", required=True)
    parser.add_argument("--dflash", required=True)
    parser.add_argument("--out")
    args = parser.parse_args()
    result = compare(args.ar, args.dflash)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(payload + "\n")
    print(payload)
    return 0 if result["match"] else 1


if __name__ == "__main__":
    sys.exit(main())
