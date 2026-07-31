# Exporting a BANANA_SMASHER model pack

The serving image never embeds the 101 GB model package. `docker/scripts/export_pack.py` creates a separately publishable directory with a fail-closed manifest.

## Inputs

You need:

- the exact 1,645-file BANANA_SMASHER plane tree;
- the mixed-tier compact overlay;
- the matching tokenizer JSON;
- enough destination capacity, or a destination on the same filesystem where hard links can be used safely.

The upstream serving receipt identifies the sealed resident source as:

```text
bytes                            101346700411
files                            1645
sealed_source_inventory_sha256   cb00fc4e783ab97018bbe0642556820596a7846816fb0bcc55bd9f27b223b3bd
```

## Build the pack

```bash
cd glm52-ds4-bq3-ptq-opd
python3 docker/scripts/export_pack.py \
  --source-root /data/banana_smasher/wire43 \
  --overlay /data/banana_smasher/mixed_tier_compact.pt \
  --tokenizer /data/banana_smasher/tokenizer.json \
  --output /data/releases/banana_smasher-mixed-tier-pack \
  --model-id deepseek-v4-mixed-tier-prefill-ladder \
  --expected-bytes 101346700411 \
  --expected-files 1645 \
  --expected-inventory-sha256 38a1b3eefaef21b6bc3368be2d4e6480a3c30d64a1e19d5239a9ac89b87233f9 \
  --sealed-source-inventory-sha256 cb00fc4e783ab97018bbe0642556820596a7846816fb0bcc55bd9f27b223b3bd
```

The output directory must be empty. On one filesystem, payload files may be hard-linked to avoid a second physical 101 GB write; treat both trees as immutable after export. Across filesystems, files are copied.

## Layout

```text
banana_smasher-mixed-tier-pack/
  MANIFEST.json
  planes/...
  overlay/mixed_tier_compact.pt
  tokenizer/tokenizer.json
```

`MANIFEST.json` uses schema `banana_smasher-pack`, version 1. It binds:

- model ID and systems-only quality scope;
- resident plane byte/file counts and a canonical pack inventory SHA-256 over
  `path NUL bytes NUL sha256 LF` rows sorted by the exact pack-relative path;
- the upstream sealed source identity separately as
  `sealed_source_inventory_sha256` (the historical receipt did not publish its
  canonicalization algorithm, so it must not be substituted for the pack hash);
- exact serving shape: 43 layers, 256 experts, top-k 6, four tiers;
- every relative payload path, role, size, and SHA-256;
- overlay and tokenizer paths.

No symlinks, path traversal, undeclared files, schema drift, byte drift, or hash drift are accepted.

## Verify before publishing

With source Python:

```bash
PYTHONPATH=docker/scripts python3 docker/scripts/entrypoint.py verify \
  /data/releases/banana_smasher-mixed-tier-pack
```

With the final image:

```bash
docker run --rm --read-only \
  -v /data/releases/banana_smasher-mixed-tier-pack:/model:ro \
  spark-bench/banana_smasher-p602:2026-07-25 verify /model
```

Both commands emit a JSON PASS receipt. Publish the pack as its own release asset or Hugging Face repository; do not commit the 101 GB plane tree to Git. A tar archive is also accepted by the serving entrypoint when provided as an HTTP(S) URL. The archive must contain exactly one `MANIFEST.json`, may not contain links, and is revalidated after extraction.
