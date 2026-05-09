#!/usr/bin/env bash
# Build the exact image family that produced the 2026-05-09 API-mode PASS.
# Run from ../vllm-prefill-flashqla-hkv-spark after building the base/current image.
set -euo pipefail

BASE_IMAGE="${BASE_IMAGE:-vllm-prefill-flashqla-hkv-spark:spark1-current}"
OUT_IMAGE="${OUT_IMAGE:-vllm-prefill-flashqla-hkv-spark:spark1-fusedo-qkgemm-alias-kpack2}"
EDIT_CONTAINER="${EDIT_CONTAINER:-flashqla-alias-kpack2-edit}"

FUSED=/opt/flashqla/flash_qla/ops/gated_delta_rule/chunk/blackwell/fused_fwd.py
SITE=/usr/lib/python3.12/sitecustomize.py

docker rm -f "$EDIT_CONTAINER" >/dev/null 2>&1 || true
docker run -d --name "$EDIT_CONTAINER" --entrypoint sleep "$BASE_IMAGE" infinity >/dev/null

docker exec "$EDIT_CONTAINER" python3 - <<'PY'
from pathlib import Path

fused = Path('/opt/flashqla/flash_qla/ops/gated_delta_rule/chunk/blackwell/fused_fwd.py')
site = Path('/usr/lib/python3.12/sitecustomize.py')

def repl(path, old, new, label):
    s = path.read_text()
    if old not in s:
        raise SystemExit(f'missing patch anchor: {label}')
    path.write_text(s.replace(old, new, 1))

# 1) Correct the fused-output QK producer: generic T.gemm with k_pack=2.
repl(fused,
'''                    T.gemm_v1(
                        q_shared[:, :],
                        k_shared[:, :],
                        p_fragment,
                        transpose_B=True,
                        clear_accum=True,
                    )''',
'''                    T.gemm(
                        q_shared[:, :],
                        k_shared[:, :],
                        p_fragment,
                        transpose_B=True,
                        clear_accum=True,
                        k_pack=2,
                    )''',
'QK gemm_v1 -> T.gemm k_pack=2')

# 2) Store the fused consumer output directly to global O instead of copying Vd into o_shared.
repl(fused,
'''                    # HYBRID: expose internal Vd/v_new in output slot.
                    # The Blackwell Pg@Vd fused-output path has a TileLang shared-layout bug;
                    # vLLM chunk_fwd_o computes the final output from (q,k,v_new,h,g) correctly.
                    T.copy(vd_shared, o_shared)''',
'''                    # Store final output directly from the consumer branch.
                    # This avoids the Blackwell shared-layout path through o_shared.
                    if store_o:
                        for j_s, j_v in T.Parallel(block_S, block_DV):
                            with T.If(seq_start_idx + i_s * block_S + j_s < seq_end_idx):
                                with T.Then():
                                    o[
                                        batch_idx,
                                        seq_start_idx + i_s * block_S + j_s,
                                        bh,
                                        bv * block_DV + j_v,
                                    ] = o_fragment[j_s, j_v]''',
'direct global output store')

# 3) Disable old producer-side o_shared store sites; output is now stored in consumer branch.
for old in [
'''                        if i_s > 0 and store_o:
                            T.copy(''',
'''                        if num_unmasked_iters > 0 and store_o:
                            T.copy(''',
'''                    if store_o:
                        for j_s, j_v in T.Parallel(block_S, block_DV):''']:
    new = old.replace('if ', 'if False and ', 1)
    repl(fused, old, new, 'disable old O store')

# 4) Route sitecustomize through FlashQLA fused output: no h output, return v_new as out.
repl(site,
'''                    cu_seqlens=cu_seqlens,output_final_state=output_final_state,output_h=True,auto_cp=False)
                out = _vllm_chunk_fwd_o(q,k,v_new,h_fq,g_cum,scale,cu_seqlens=cu_seqlens,chunk_indices=chunk_indices)''',
'''                    cu_seqlens=cu_seqlens,output_final_state=output_final_state,output_h=False,auto_cp=False)
                out = v_new''',
'sitecustomize direct fused output')
PY

docker commit "$EDIT_CONTAINER" "$OUT_IMAGE" >/dev/null
docker rm -f "$EDIT_CONTAINER" >/dev/null

echo "built $OUT_IMAGE from $BASE_IMAGE"
echo "sha256s:"
docker run --rm --entrypoint bash "$OUT_IMAGE" -lc "sha256sum $SITE $FUSED /opt/flashqla_hkv_o.py"
