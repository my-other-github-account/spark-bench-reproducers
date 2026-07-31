# Release-candidate preseal

Status: `RELEASE_CANDIDATE_PENDING_FULL_PACK_GATE` means the source/build contract is public, but no immutable image digest or measured full-pack gate receipt has been accepted. It is not a golden or performance claim.

A **bs-pack v1 artifact** is a model directory governed by version 1 of [`PACK_FORMAT.md`](PACK_FORMAT.md). **Fail-closed validation** means any missing, extra, symlinked, resized, or hash-drifted pack row exits nonzero before model allocation.

## Build from the public clone

Run the three commands in [`README.md`](README.md) from a clean public clone pinned to the release-candidate commit. The build must leave the stock image contract unchanged: null entrypoint and default command `vllm serve /model`.

## Validate before launch

Place the complete artifact at `$PWD/bs-pack-v1`, then run the image-contained validator before any server launch:

```text
PACK="$PWD/bs-pack-v1"
docker run --rm -v "$PACK:/model:ro" banana_smasher-serve:golden smash validate-pack /model
```

A nonzero exit is terminal for this candidate. Do not launch, benchmark, or relabel the image after a validation failure.

## Launch stock vLLM

The following uses only standard `vllm serve` flags; banana-smasher adds no launcher, wrapper, required environment variable, or serving objective:

```text
docker run --rm --name banana-smasher --gpus all --ipc=host \
  -v "$PACK:/model:ro" -p 8000:8000 banana_smasher-serve:golden \
  vllm serve /model --host 0.0.0.0 --port 8000 \
  --max-model-len 8192 --max-num-seqs 16
```

Wait until both `/health` and `/v1/models` return HTTP 200.

## Three-output eyeball gate

The **three-output eyeball gate** is a pre-benchmark sanity check: request exactly three deterministic greedy outputs, then inspect all three for nonempty text, instruction/language relevance, and absence of obvious corruption or runaway repetition. It is not a quality score and cannot promote the candidate.

```text
python3 - <<'PY'
import json, urllib.request
base = "http://127.0.0.1:8000"
model = json.load(urllib.request.urlopen(base + "/v1/models"))["data"][0]["id"]
prompts = [
    "Answer with only the number: 7 + 5 =",
    "Write one grammatical sentence about a banana.",
    "Return the lowercase word ready and nothing else.",
]
for index, prompt in enumerate(prompts, 1):
    body = json.dumps({"model": model, "prompt": prompt, "max_tokens": 32,
                       "temperature": 0, "seed": 1}).encode()
    request = urllib.request.Request(base + "/v1/completions", body,
                                     {"Content-Type": "application/json"})
    result = json.load(urllib.request.urlopen(request))
    print(f"OUTPUT_{index}: {result['choices'][0]['text']!r}")
PY
```

Stop on any failed check. Preserve the three outputs in the later measured gate receipt; do not infer a performance claim from them.

## C1/C2/C4/C8/C16 ladder

A **C-shape** `Cn` is one measured batch with exactly `n` concurrent requests. The **C1/C2/C4/C8/C16 ladder** is the ordered set of those five shapes, run against one unchanged image, pack, server argv, prompt-token count, output-token count, and measurement formula.

For each shape in order, use stock `vllm bench serve` against `http://127.0.0.1:8000`: run one warm-up batch and exclude it, then run three measured batches with `--request-rate inf`, `--num-prompts n`, and `--max-concurrency n`. Record raw rows and the median for every shape. Aggregate throughput is total completion tokens divided by batch wall time. Do not skip C8 or C16, substitute decode-only timing, mix configurations, or report a golden/performance result from this release candidate.

The full-pack gate receipt must bind the validated pack manifest, immutable image digest, stock server argv, three eyeball outputs, all raw ladder rows, all five medians, warm-up exclusion, and the aggregate formula. This repository deliberately contains no provisional `RELEASE.json`.

## Two-field golden seal

After the independent full-pack gate passes, run `tools/seal_release.py` with exactly two late-bound facts: the real immutable `sha256:` image digest and the measured PASS gate receipt. The helper rejects malformed digests, non-PASS receipts, receipts that do not bind the same digest, receipts without a pack-manifest SHA-256, or receipts without all five ladder shapes. It writes a deterministic `RELEASE.json` and never hard-codes a fake digest or performance value.
