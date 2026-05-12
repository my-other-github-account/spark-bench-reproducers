#!/usr/bin/env bash
# Apply every patch needed to reproduce the one-serve composition on a clean
# DGX Spark (GB10 sm_121a) with vLLM nightly + FlashInfer 0.6.11.
#
# Run order:
#  1. AEON vLLM source patches (5 patches into installed vllm dist-package)
#  2. FlashQLA build + install with HKV V1 patch + sitecustomize hook
#  3. Codex GDN qkv/z dynamic FP4 patch into vllm GDN linear-attn layer
#
# Idempotent: each step checks for prior application.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-/home/user/venvs/vllm}"
PY_VER="${PY_VER:-python3.12}"
SITE_PACKAGES="${SITE_PACKAGES:-$VENV_DIR/lib/$PY_VER/site-packages}"
VLLM_PKG="${VLLM_PKG:-$SITE_PACKAGES/vllm}"

echo "=== Setup environment ==="
echo "  REPO_ROOT:     $REPO_ROOT"
echo "  VENV_DIR:      $VENV_DIR"
echo "  SITE_PACKAGES: $SITE_PACKAGES"
echo "  VLLM_PKG:      $VLLM_PKG"

if [ ! -d "$VLLM_PKG" ]; then
  echo "ERROR: vLLM not found at $VLLM_PKG. Install vLLM nightly first."
  exit 1
fi

# ---------------------------------------------------------------
# Step 1 — AEON vLLM source patches (5 patches into installed vllm)
# ---------------------------------------------------------------
echo ""
echo "=== Step 1: AEON vLLM source patches ==="
cd "$REPO_ROOT/patches/aeon_vllm"

# Apply in order. Each is idempotent — prints "already applied" if upstream landed it.
"$VENV_DIR/bin/python" register_qwen3_5_text.py        || echo "  (register_qwen3_5_text already applied or upstream)"
"$VENV_DIR/bin/python" patch_cuda_optional_import.py   || echo "  (patch_cuda_optional_import already applied)"
"$VENV_DIR/bin/python" patch_kv_cache_utils.py         || echo "  (patch_kv_cache_utils already applied)"
"$VENV_DIR/bin/python" patch_mrope_text_fallback.py    || echo "  (patch_mrope_text_fallback already applied)"
"$VENV_DIR/bin/python" patch_cudagraph_align.py        || echo "  (patch_cudagraph_align already applied)"

# Optional: PR40898 DFlash SWA overlay (only if your vLLM nightly predates merge)
# "$VENV_DIR/bin/python" apply_pr40898_dflash_swa.py

# ---------------------------------------------------------------
# Step 2 — FlashQLA + HKV V1 + sitecustomize hook
# ---------------------------------------------------------------
echo ""
echo "=== Step 2: FlashQLA HKV V1 install ==="

FLASHQLA_DIR="${FLASHQLA_DIR:-/opt/flashqla}"
FLASHQLA_COMMIT="827fdd88e0829646e3c90be0c76158a9be62ab37"

if [ ! -d "$FLASHQLA_DIR/.git" ]; then
  echo "  Cloning FlashQLA at pinned commit..."
  sudo git clone https://github.com/my-other-github-account/FlashQLA.git "$FLASHQLA_DIR"
  sudo chown -R "$USER" "$FLASHQLA_DIR"
fi

cd "$FLASHQLA_DIR"
git fetch --quiet origin
git checkout "$FLASHQLA_COMMIT"

# Apply the HKV V1 source patch (idempotent: skip if already applied)
HKV_PATCH="$REPO_ROOT/patches/flashqla_hkv_v1/flashqla-source-diff-827fdd88-hkv.patch"
if ! git apply --reverse --check "$HKV_PATCH" >/dev/null 2>&1; then
  git apply "$HKV_PATCH" || { echo "  HKV patch already in source tree (or conflict)"; }
else
  echo "  (HKV patch already applied)"
fi

# Install runtime deps required by the HKV Triton kernel
"$VENV_DIR/bin/pip" install --no-cache-dir \
  "tilelang==0.1.8" "apache-tvm-ffi==0.1.9" "flash_linear_attention==0.5.0"

# Install FlashQLA into venv
cd "$FLASHQLA_DIR"
"$VENV_DIR/bin/pip" install --no-cache-dir -e . --no-build-isolation

# Drop the HKV output kernel + sitecustomize into /opt and the venv
sudo cp "$REPO_ROOT/patches/flashqla_hkv_v1/flashqla_hkv_o.py" /opt/flashqla_hkv_o.py
sudo chown "$USER" /opt/flashqla_hkv_o.py

# sitecustomize.py — copy or merge with existing
SITE_CUST="$SITE_PACKAGES/sitecustomize.py"
SRC_SITE="$REPO_ROOT/patches/flashqla_hkv_v1/sitecustomize.py"
if [ -f "$SITE_CUST" ] && ! grep -q "flashqla-patch" "$SITE_CUST"; then
  echo "  Existing sitecustomize.py found — backing up and merging."
  cp "$SITE_CUST" "$SITE_CUST.bak.$(date +%s)"
  cat "$SRC_SITE" >> "$SITE_CUST"
elif [ ! -f "$SITE_CUST" ]; then
  cp "$SRC_SITE" "$SITE_CUST"
else
  echo "  (sitecustomize.py already has [flashqla-patch])"
fi

# ---------------------------------------------------------------
# Step 3 — Codex GDN qkv/z dynamic FP4 patch
# ---------------------------------------------------------------
echo ""
echo "=== Step 3: Codex GDN qkv/z dynamic FP4 patch ==="

# Two-file patch — applies to gdn_linear_attn.py + model_loader_utils.py
# The successful run used qkv/z-ONLY variant (reverted from qkv/z+b/a — see Critical Notes in README).
GDN_PATCHES_DIR="$REPO_ROOT/patches/codex_gdn_qkvz_fp4"

cd "$VLLM_PKG"

# Apply the v2 (qkv/z-only) GDN + loader patches.
# Order matters: GDN patch first, then loader patch.
for p in \
  "$GDN_PATCHES_DIR/codex_qwen36_gdn_qkvz_dynamic_fp4_v2_gdn_20260511_010436.diff" \
  "$GDN_PATCHES_DIR/codex_qwen36_gdn_qkvz_dynamic_fp4_v2_loader_20260511_010436.diff" \
  "$GDN_PATCHES_DIR/codex_qwen36_gdn_qkvz_dynamic_fp4_importfix_20260511_010628.diff"
do
  if [ ! -f "$p" ]; then continue; fi
  pname="$(basename "$p")"
  if patch --dry-run --reverse -p1 -F0 < "$p" >/dev/null 2>&1; then
    echo "  (already applied: $pname)"
  else
    echo "  Applying $pname"
    patch -p1 < "$p"
  fi
done

# ---------------------------------------------------------------
# Verify
# ---------------------------------------------------------------
echo ""
echo "=== Verification ==="
"$VENV_DIR/bin/python" - <<'PY'
import importlib.metadata as md, sys
for pkg in ("vllm", "flashinfer-python", "torch", "tilelang", "flash_linear_attention"):
    try:
        v = md.version(pkg)
        print(f"  {pkg}: {v}")
    except md.PackageNotFoundError:
        print(f"  {pkg}: NOT INSTALLED")

# Confirm AEON patches landed
import pathlib
markers = {
  "cudagraph_align_spec_decode_all_modes": "vllm/config/compilation.py",
  "kv_cache_utils_min_none_safe":          "vllm/v1/core/kv_cache_utils.py",
  "mrope_text_fallback":                   "vllm/v1/worker/gpu_model_runner.py",
  "stable_libtorch_lazy_dlopen":           "vllm/platforms/cuda.py",
}
import vllm
vllm_root = pathlib.Path(vllm.__file__).parent
for marker, rel in markers.items():
    p = vllm_root / pathlib.Path(rel.replace("vllm/", ""))
    if marker in p.read_text():
        print(f"  AEON marker {marker}: OK")
    else:
        print(f"  AEON marker {marker}: MISSING in {p}")

# Codex GDN patch marker
gdn = vllm_root / "model_executor/layers/mamba/gdn_linear_attn.py"
if "CODEX_QWEN36_GDN_QKVZ_DYNAMIC_FP4" in gdn.read_text():
    print("  Codex GDN qkv/z FP4 marker: OK")
else:
    print("  Codex GDN qkv/z FP4 marker: MISSING")

# FlashQLA install
try:
    import flashqla
    print(f"  FlashQLA imported: {flashqla.__file__}")
except ImportError as e:
    print(f"  FlashQLA NOT imported: {e}")

# HKV output kernel reachable
sys.path.insert(0, "/opt")
try:
    import flashqla_hkv_o
    print(f"  flashqla_hkv_o imported: {flashqla_hkv_o.__file__}")
except ImportError as e:
    print(f"  flashqla_hkv_o NOT imported: {e}")
PY

echo ""
echo "=== Setup complete ==="
echo "Next: bash scripts/launch_one_serve.sh"
