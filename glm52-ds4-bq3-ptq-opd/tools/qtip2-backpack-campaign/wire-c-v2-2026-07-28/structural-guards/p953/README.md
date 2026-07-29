# P953 immutable-SHA and resume guard reference

This directory publishes the ready-to-adopt P953 guard module and a privacy-safe standalone regression suite.

Authority status:

- `immutable_sha_authority.py` is byte-identical to the sealed ready-to-adopt module.
- Source/module SHA-256: `e2d9d1dbbaa06fb45bfb6b0aba693f77eed3ff1163d3e1aa15f5373f5fc4078c`.
- The source seal recorded 11/11 passing mission regressions and status `PASS_READY_TO_ADOPT_NOT_DEPLOYED`.
- This public directory does not claim that the later canonical-integration successor passed review. It is a reference guard and negative-test surface, not proof of fleet deployment.

Run the public regressions:

```bash
python3.13 test_immutable_sha_and_resume.py
```

The tests prove fail-closed behavior for missing, duplicate, wrong-byte, and index-SHA cases; path/name invariance; exact hash-bound inherited prefixes; L000-L013 resume beginning at L014; and refusal to recompute when the checkpoint sidecar is absent.
