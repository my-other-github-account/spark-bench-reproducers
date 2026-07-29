# Structural authority guards

This package makes artifact identity structural instead of path-dependent.

Core invariants:

1. Immutable payloads live at `authority_store/store/<sha256>.bin`. Mission paths are provenance only.
2. Builders and integrators resolve a sealed plan's expected SHA through `AuthorityStore`; they never fall back to a historical path.
3. A SHA substitution is rejected unless `SUBSTITUTION_WAIVER.json` is adjacent to the plan, contains exactly the schema's six measured fields, matches the exact expected/substitute pair, reports a finite numeric measured delta and 95% interval from at least 64 windows, and references a byte-present measurement receipt in the authority store.
4. `build-index` examines every JSON/JSONL document below its explicit mission roots (no filename allowlist) and extracts SHA references from PASS/SEALED manifests, including K1/layer archives. `reclaim-check` rejects any protected file before deletion unless a PASS `ARCHIVE_FIRST.json` proves an exact on-disk NAS readback copy through `nas_path` + `readback_sha256`.
5. `seal-check` full-reads every declared dependency on at least two distinct hosts. A one-host dependency cannot seal.

Examples:

```bash
python3 -m authority.cli ingest --root $AUTHORITY_ROOT path/to/codebook.bin
python3 -m authority.cli plan-resolve --root $AUTHORITY_ROOT --plan PLAN.json --row ROW.json
python3 -m authority.cli build-index --output $AUTHORITY_ROOT/protected_sha_index.jsonl sealed/*.json
python3 -m authority.cli reclaim-check --index $AUTHORITY_ROOT/protected_sha_index.jsonl candidate_dir
python3 -m authority.cli seal-check --dependencies deps.json --locations authority_locations.json --output copy_census.json
```

`authority_locations.json` supports local and SSH probes:

```json
{
  "host-a": {"mode": "local", "root": "/srv/authority-store"},
  "host-b": {"mode": "ssh", "target": "PUBLIC_OPERATOR@PUBLIC_NODE_ADDRESS", "root": "/srv/authority-store"}
}
```

SSH probes are read-only. Store replication remains an explicit operator step; seal fails closed until all required copies exist.
