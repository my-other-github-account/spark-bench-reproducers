# Publication scrub policy

This campaign repository is public. Reproduction material must describe roles and inputs without publishing private infrastructure.

Required before every push:

```bash
python3 tools/scrub_audit.py
```

The audit fails on:

- private or overlay-network IPv4 literals;
- the campaign operator's absolute Linux home path;
- workstation or mDNS hostnames;
- shorthand fleet names other than `spark-N` in documentation and data;
- common personal-access-token, cloud-key, and private-key signatures.

Use these public placeholders:

- `$MISSION_ROOT` for experiment state and receipts;
- `$MODEL_ROOT` for the source checkpoint;
- `$CORPUS_ROOT` for pinned evaluation corpora;
- `spark-N` only when host identity is relevant, otherwise `$BUILD_HOST`, `$EVAL_HOST`, and `$SERVE_HOST`;
- `$FABRIC_IFACE` and `$HEAD_ADDR` rather than addresses.

The audit is a leak detector, not a guarantee. Review staged diffs before pushing and never commit environment files, credentials, model weights, activation caches, or raw user data.
