# Structural guard bundle

This directory publishes the canonical TRUE-C/F521 immutable-SHA and resumable-scorer guard together with two independently useful reference surfaces:

- Top level: byte-identical canonical guard named in `CANONICAL_SHA256.json`. It enforces expected-SHA selection, duplicate rejection, source-byte hashing, contiguous receipt chains, and restart at the first unfinished layer. `MEASUREMENT_CARD_TEMPLATE.md` makes expected-SHA manifest binding mandatory for future measurement cards.
- `p936/`: append-only SHA authority store, protected-SHA reverse index, archive-first reclaim refusal, measured substitution waiver, two-node dependency census, CLI integrations, schemas, and executable unit/negative tests.
- `p953/`: a compact ready-to-adopt immutable-SHA selection and resumable-layer reference module with privacy-safe negative regressions.

The top-level module is the canonical production-identical source. The nested bundles retain the audited historical guard surfaces and are not represented as separately fleet-deployed integrations.

Run all three surfaces with Python 3.13 or newer:

```bash
python3 test_structural_guards.py
(cd p936 && python3 -m unittest discover -s authority/tests -v)
(cd p953 && python3 test_immutable_sha_and_resume.py)
```

The tests are intentionally negative-heavy: they remove provenance paths while retaining expected-SHA authority objects, reject missing/duplicate/wrong objects, enforce exact contiguous completed-layer prefixes, and reject tampered sufficient statistics.
