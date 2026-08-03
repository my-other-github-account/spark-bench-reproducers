# Banana Smasher runtime

This standalone repository owns one path only:

materialized quant source -> `smash export` -> `smash verify` -> self-contained `/model` pack -> pinned stock vLLM image with `vllm.general_plugins` -> OpenAI-compatible API.

It does not contain training, solver orchestration, benchmark ledgers, or historical run artifacts.

## Build the Python packages

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip build
python -m pip install -e ./banana-smasher
```

The runtime image builds and installs both `banana-smasher` and `banana-smasher-plugin` from this checkout. The plugin is discovered from the `vllm.general_plugins` entry point; no `PYTHONPATH` overlay is used.

## Export and verify a model pack

`RUNTIME_FLOOR_BYTES` is mandatory for the P1016 path. Set it to the measured value from the caller-provided runtime receipt; this repository deliberately provides no default.

```bash
export QUANT_SOURCE=/path/to/materialized-quant-source
export SERVING_MODEL=/path/to/serveable-base-model
export MODEL_OUT=/path/to/model-pack
export MODEL_ID=your-model-id
export INSTANCE_ID=your-pack-instance
export RUNTIME_FLOOR_BYTES=MEASURED_RECEIPT_VALUE

smash export --source-root "$QUANT_SOURCE" --runtime-floor-bytes "${RUNTIME_FLOOR_BYTES:?required from a measured receipt}" --serving-model-root "$SERVING_MODEL" --output "$MODEL_OUT" --model-id "$MODEL_ID" --instance-id "$INSTANCE_ID" --link-mode copy
smash verify "$MODEL_OUT"
```

`examples/export_model.sh` provides the same fail-closed command. The resulting directory is self-contained and is the only directory mounted at `/model` for serving.

## Build and serve

```bash
docker build --file docker/Dockerfile --tag banana-smasher-runtime:local .
docker run --rm --gpus all -p 8000:8000 -v "$MODEL_OUT:/model:ro" banana-smasher-runtime:local
```

The image's exact `CMD` runs `vllm serve /model` with the pinned runtime defaults. Do not replace it with an alternate launcher.

## OpenAI API smoke test

```bash
python examples/smoke_api.py
curl -sS http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"banana-smasher-v5","messages":[{"role":"user","content":"Reply with OK."}],"max_tokens":8}'
```

## Verification surfaces

- `ACCELERATIONS.md` explains every retained acceleration.
- `runtime/ACCELERATION_MANIFEST.json` is the machine-readable source/build/activation/test map.
- `PROVENANCE.md` and `provenance/SOURCE_INVENTORY.json` bind retained files to source commit `c00714c6803f7e2de7a95d103dbe172236b22adf`.
- `python -m pytest -q` runs package, plugin-feasible, Docker-static, and extraction-contract tests.

The CUDA image build and GPU boot require a Linux ARM64 CUDA host with an SM120/SM121 GPU. They are a separate hardware gate; a macOS test pass is not evidence of a GPU boot.
