#!/bin/bash
# fix-dflash-off-by-one: shift DFlash drafter target_layer_ids by +1 so vLLM's
# forward loop captures the outputs of the transformer layers the drafter was
# trained against. See README.md for the full story.
set -e

VLLM_DIR="${VLLM_SITE_PACKAGES:-/usr/local/lib/python3.12/dist-packages/vllm}"
TARGET="${VLLM_DIR}/v1/worker/gpu_model_runner.py"

if [[ ! -f "$TARGET" ]]; then
    echo "fix-dflash-off-by-one: cannot find $TARGET — skipping."
    exit 0
fi

if grep -q "DFlash layer-tap off-by-one fix applied" "$TARGET"; then
    echo "fix-dflash-off-by-one: already applied, skipping."
    exit 0
fi

# Apply in-place: locate the dflash_config branch and shift layer ids.
python3 - <<'PY'
import io, re, pathlib, sys, os
target = pathlib.Path(os.environ.get("VLLM_SITE_PACKAGES",
    "/usr/local/lib/python3.12/dist-packages/vllm")) / "v1/worker/gpu_model_runner.py"
src = target.read_text()

# Guard: skip if something equivalent is already present.
if "DFlash layer-tap off-by-one fix applied" in src:
    print("already patched"); sys.exit(0)

# Anchor: the block that reads `layer_ids = dflash_config.get("target_layer_ids")`
# and then falls through to `if layer_ids and isinstance(layer_ids, (list, tuple)):`.
anchor = 'layer_ids = dflash_config.get("target_layer_ids")\n'
if anchor not in src:
    print("fix-dflash-off-by-one: anchor not found, vLLM version may be too old/new — skipping")
    sys.exit(0)

# Introduce a dflash_source flag when layer_ids come from dflash_config.
old1 = '''            if dflash_config and isinstance(dflash_config, dict):
                layer_ids = dflash_config.get("target_layer_ids")
'''
new1 = '''            if dflash_config and isinstance(dflash_config, dict):
                layer_ids = dflash_config.get("target_layer_ids")
                _dflash_source = bool(layer_ids)
'''
if old1 not in src:
    # Already marked (re-patch) — defensive fallback.
    _dflash_source_introduced = True
else:
    src = src.replace(old1, new1, 1)
    _dflash_source_introduced = False

# Insert the +1 shift immediately before the final `return` of the layer_ids tuple.
old2 = "        if layer_ids and isinstance(layer_ids, (list, tuple)):\n            return tuple(layer_ids)\n"
new2 = ("        if layer_ids and isinstance(layer_ids, (list, tuple)):\n"
        "            if locals().get('_dflash_source'):\n"
        "                # DFlash layer-tap off-by-one fix (mods/fix-dflash-off-by-one):\n"
        "                # reference DFlash reads HF hidden_states tuple with offset=1\n"
        "                # (dflash/model.py::extract_context_feature); vLLM's forward loop\n"
        "                # captures the OUTPUT of layer N when layer_idx=N+1 is requested\n"
        "                # (layer_idx=0 is pre-embedding). Translate reference semantics\n"
        "                # -> vLLM semantics by adding 1.\n"
        "                layer_ids = tuple(int(l) + 1 for l in layer_ids)\n"
        "                import logging as _logging\n"
        "                _logging.getLogger('vllm').info(\n"
        "                    'DFlash layer-tap off-by-one fix applied: aux_hidden_state_layers=%s',\n"
        "                    layer_ids,\n"
        "                )\n"
        "            return tuple(layer_ids)\n")
if old2 not in src:
    print("fix-dflash-off-by-one: return-anchor not found, skipping"); sys.exit(0)
src = src.replace(old2, new2, 1)

target.write_text(src)
print("fix-dflash-off-by-one: applied")
PY

echo "fix-dflash-off-by-one: done"
