# Structural guard bundle

Two independently useful guard surfaces are published here:

- `p936/`: append-only SHA authority store, protected-SHA reverse index, archive-first reclaim refusal, measured substitution waiver, two-node dependency census, CLI integrations, schemas, and 12 executable unit/negative tests.
- `p953/`: exact ready-to-adopt immutable-SHA selection and resumable-layer reference module plus privacy-safe negative regressions.

These are executable reference guards. P936 is the complete source/test bundle recovered from its sealed patch. P953 is explicitly labeled ready-to-adopt/not-deployed; this package does not misrepresent an unreviewed canonical integration as fleet-deployed.

Run:

```bash
(cd p936 && python3 -m unittest discover -s authority/tests -v)
(cd p953 && python3.13 test_immutable_sha_and_resume.py)
```
