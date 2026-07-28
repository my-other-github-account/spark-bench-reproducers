# Immutable-SHA and resumable scorer guards

This directory is the public, executable reproduction of the canonical TRUE-C/F521 structural guards.

The exact shared guard module is byte-identical to the production module named in `CANONICAL_SHA256.json`. It enforces:

- expected-SHA selection from a sealed, duplicate-free authority index;
- source-byte hashing before use, independent of provenance paths and human-readable names;
- fail-closed missing, duplicate, byte-count, and content-SHA behavior;
- exact contiguous completed-layer prefixes;
- immutable per-layer receipt chains with sufficient-statistics hashes;
- restart at the first unfinished layer without fallback recomputation.

Python 3.9 or newer is supported. Run:

```bash
python3 test_structural_guards.py
```

The tests are intentionally negative-heavy. They delete the provenance path while retaining the expected-SHA authority object, reject missing/duplicate/wrong objects, preseed L000-L013 and require L014 as the first pending layer, and reject tampered completed-layer statistics.

`MEASUREMENT_CARD_TEMPLATE.md` is mandatory for future measurement cards. A card without an expected-SHA manifest binding is not runnable.
