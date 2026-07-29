"""F521 integration hook: replace mission paths with authority SHA bindings."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from ..authority_guard import AuthorityStore, GuardViolation, resolve_plan_codebook


def resolve_f521_codebooks(
    codebook_receipt_path: os.PathLike[str] | str,
    plan_path: os.PathLike[str] | str,
    authority_store_root: os.PathLike[str] | str,
) -> dict[str, Path]:
    """Resolve every F521 receipt codebook from ``store/<sha>.bin``.

    Receipt ``path`` values are provenance only and are deliberately ignored.
    """
    receipt_path = Path(codebook_receipt_path).expanduser().resolve()
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise GuardViolation("invalid F521 codebook receipt") from exc
    rows = receipt.get("rows") if isinstance(receipt, Mapping) else None
    if not isinstance(rows, list) or not rows:
        raise GuardViolation("F521 codebook receipt has no rows")
    store = AuthorityStore(authority_store_root)
    resolved: dict[str, Path] = {}
    byte_counts: dict[str, int] = {}
    for specification in rows:
        if not isinstance(specification, Mapping):
            raise GuardViolation("invalid F521 codebook row")
        path = resolve_plan_codebook(store, plan_path, {"codebook": specification})
        digest = str(specification["sha256"])
        expected_bytes = int(specification["bytes"])
        if digest in byte_counts and byte_counts[digest] != expected_bytes:
            raise GuardViolation("F521 codebook SHA has conflicting byte counts")
        byte_counts[digest] = expected_bytes
        resolved[digest] = path
    return resolved
