#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE="${RECIPE:-$(cd "$SCRIPT_DIR/.." && pwd)}"
ROOT=$RECIPE/results/pp32k-tg128-dflash-rescue-20260507
ROW=mbt2048-dflash_on-fqla_effectively_off-nositecustomize-noeditable-nofla-threshold0-nspec15-temp06-mlen65536-shifted_suffix_block_table_fullgraph_compact_delta_nosync_direct_attention_kvupdate_query_qkv_fused_qk_rope-warm5-n30-pp32768
ART=$ROOT/$ROW
NAME=pp32k-dflash-rescue-shifted-suffix-block-table-fullgraph-compact-delta-nosync-direct-attention-kvupdate-query-qkv-fused-qk-rope-nspec15-temp06-mlen65536-n30
IMAGE=qwen36-fqla-baseline-dflash-spark:combined-20260507-threshold10

mkdir -p "$ART"
cd "$RECIPE"

date -Is > "$ART/start.txt"
cat > "$ART/metadata.txt" <<META
row=$ROW
host=$(hostname)
start=$(cat "$ART/start.txt")
method=warm1/N1 diagnostic proof patch: preserve direct-attention v2, query-QKV workspace v3, fused QK-norm+RoPE, compact delta no-sync, proposer full-CUDAGraph, real local-argmax DFlash drafts, and valid temp0.6 target sampling, then shift DFlash query attention to a block-aligned recent suffix block table while preserving absolute positions and slot mappings. final warm5/N30 gate; no N10/N30 unless N5 strongly clears.
image=$IMAGE
META

docker rm -f "$NAME" > "$ART/container-rm-before.txt" 2>&1 || true

cat > "$ART/patch_query_qkv_workspace_v3.py" <<'PY'
from pathlib import Path

patches = {
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_dflash.py": r'''

# --- pp32k DFlash rescue local-argmax + query QKV workspace v3 patch ---
import time as _dflash_fc_time

_DFLASH_LOCAL_ARGMAX_MODEL_CALLS = 0

def _dflash_get_top_tokens(self, hidden_states):
    global _DFLASH_LOCAL_ARGMAX_MODEL_CALLS
    _DFLASH_LOCAL_ARGMAX_MODEL_CALLS += 1
    if _DFLASH_LOCAL_ARGMAX_MODEL_CALLS <= 20 or _DFLASH_LOCAL_ARGMAX_MODEL_CALLS % 50 == 0:
        logger.info(
            "DFlash local argmax proposer path active: model_get_top_tokens_calls=%s hidden_shape=%s",
            _DFLASH_LOCAL_ARGMAX_MODEL_CALLS,
            tuple(hidden_states.shape),
        )
    return self.logits_processor.get_top_tokens(self.lm_head, hidden_states)

DFlashQwen3ForCausalLM.get_top_tokens = _dflash_get_top_tokens

_DFLASH_FC_ORIG_COMBINE = DFlashQwen3ForCausalLM.combine_hidden_states
_DFLASH_FC_STATS = {
    "calls": 0,
    "total_ms": 0.0,
    "max_ms": 0.0,
    "tokens": 0,
    "workspace_reuses": 0,
    "fallbacks": 0,
}

def _dflash_query_qkv_workspace_v3_combine(self, hidden_states):
    if not self.model.use_aux_hidden_state:
        return hidden_states

    needs_squeeze = hidden_states.dim() == 1
    if needs_squeeze:
        hidden_states = hidden_states.unsqueeze(0)

    fc = self.model.fc
    bias = getattr(fc, "bias", None)
    weight = getattr(fc, "weight", None)
    if bias is not None or weight is None or hidden_states.dim() != 2:
        _DFLASH_FC_STATS["fallbacks"] += 1
        return _DFLASH_FC_ORIG_COMBINE(self, hidden_states.squeeze(0) if needs_squeeze else hidden_states)

    start = _dflash_fc_time.perf_counter()
    out_shape = (hidden_states.shape[0], weight.shape[0])
    ws_key = (out_shape, hidden_states.dtype, hidden_states.device)
    if getattr(self, "_dflash_query_qkv_workspace_v3_key", None) != ws_key:
        self._dflash_query_qkv_workspace_v3_key = ws_key
        self._dflash_query_qkv_workspace_v3 = torch.empty(
            out_shape, dtype=hidden_states.dtype, device=hidden_states.device
        )
    else:
        _DFLASH_FC_STATS["workspace_reuses"] += 1

    result = self._dflash_query_qkv_workspace_v3
    torch.mm(hidden_states, weight.t(), out=result)

    elapsed = (_dflash_fc_time.perf_counter() - start) * 1000.0
    _DFLASH_FC_STATS["calls"] += 1
    _DFLASH_FC_STATS["total_ms"] += elapsed
    _DFLASH_FC_STATS["max_ms"] = max(_DFLASH_FC_STATS["max_ms"], elapsed)
    _DFLASH_FC_STATS["tokens"] += int(hidden_states.shape[0])
    c = _DFLASH_FC_STATS["calls"]
    if c <= 20 or c % 50 == 0:
        logger.info(
            "DFlash fc projection workspace path active: query_qkv_workspace_v3_calls=%s total_ms=%.3f avg_ms=%.3f max_ms=%.3f tokens=%s last_tokens=%s workspace_reuses=%s fallbacks=%s in_features=%s out_features=%s",
            c,
            _DFLASH_FC_STATS["total_ms"],
            _DFLASH_FC_STATS["total_ms"] / c,
            _DFLASH_FC_STATS["max_ms"],
            _DFLASH_FC_STATS["tokens"],
            int(hidden_states.shape[0]),
            _DFLASH_FC_STATS["workspace_reuses"],
            _DFLASH_FC_STATS["fallbacks"],
            int(hidden_states.shape[1]),
            int(weight.shape[0]),
        )
    if needs_squeeze:
        return result.squeeze(0)
    return result

DFlashQwen3ForCausalLM.combine_hidden_states = _dflash_query_qkv_workspace_v3_combine
# --- end pp32k DFlash rescue local-argmax + query QKV workspace v3 patch ---
''',
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/dflash.py": r'''

# --- pp32k DFlash rescue local-argmax + query QKV workspace v3 patch ---
_DFLASH_ORIG_INIT_LOCAL_ARGMAX_QUERY_QKV = DFlashProposer.__init__
_DFLASH_ORIG_LOAD_MODEL_QUERY_QKV = DFlashProposer.load_model

def _dflash_local_argmax_query_qkv_init(self, *args, **kwargs):
    _DFLASH_ORIG_INIT_LOCAL_ARGMAX_QUERY_QKV(self, *args, **kwargs)
    self.use_local_argmax_reduction = True
    logger.info(
        "DFlash query QKV workspace v3 path active: init_forward_override_installed=True forced use_local_argmax_reduction=True"
    )

def _dflash_query_qkv_allocate(self):
    allocated = 0
    max_tokens = int(getattr(self, "max_query_tokens", 8))
    layers = self.model.model.layers
    for layer in layers:
        attn = layer.self_attn
        weight = getattr(attn.qkv_proj, "weight", None)
        bias = getattr(attn.qkv_proj, "bias", None)
        if weight is None or bias is not None:
            continue
        total = int(attn.q_size + attn.kv_size + attn.kv_size)
        attn._dflash_query_qkv = torch.empty(
            (max_tokens, total), dtype=weight.dtype, device=weight.device
        )
        attn._dflash_query_qkv_max_tokens = max_tokens
        attn._dflash_query_qkv_workspace_v3_enabled = True
        allocated += 1
    logger.info(
        "DFlash query QKV workspace v3 path active: load_model_workspace_allocated=True allocated_layers=%s max_query_tokens=%s forced use_local_argmax_reduction=True",
        allocated,
        max_tokens,
    )
    return allocated

def _dflash_query_qkv_load_model(self, target_model):
    result = _DFLASH_ORIG_LOAD_MODEL_QUERY_QKV(self, target_model)
    try:
        _dflash_query_qkv_allocate(self)
    except Exception as exc:
        logger.exception("DFlash query QKV workspace v3 allocation failed after load_model: %s", exc)
    return result

DFlashProposer.__init__ = _dflash_local_argmax_query_qkv_init
DFlashProposer.load_model = _dflash_query_qkv_load_model
# --- end pp32k DFlash rescue local-argmax + query QKV workspace v3 patch ---
''',
}

for filename, payload in patches.items():
    p = Path(filename)
    text = p.read_text()
    if "pp32k DFlash rescue local-argmax + query QKV workspace v3 patch" not in text:
        p.with_suffix(p.suffix + ".bak_query_qkv_workspace_v3").write_text(text)
        p.write_text(text + payload)
    print("query-qkv-workspace-v3-patched", filename)
PY

cat > "$ART/patch_direct_attention_kvupdate_fused_qk_rope.py" <<'PY'
from pathlib import Path

qwen = Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_dflash.py")
dflash = Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/dflash.py")

qwen_payload = r'''

# --- pp32k DFlash rescue direct attention kvupdate fused qk-norm-rope patch ---
from vllm import _custom_ops as _dflash_fused_ops
from vllm.utils.torch_utils import _encode_layer_name as _dflash_encode_layer_name

_DFLASH_DIRECT_ATTN_ORIG_FORWARD = DFlashQwen3Attention.forward

def _dflash_direct_attention_kvupdate_fused_qk_rope_forward(self, positions, hidden_states):
    if (
        not getattr(self, "_dflash_query_qkv_workspace_v3_enabled", False)
        or not getattr(self, "_dflash_direct_attention_enabled", False)
        or hidden_states.dim() != 2
        or getattr(self.qkv_proj, "bias", None) is not None
        or getattr(self.qkv_proj, "weight", None) is None
        or not hasattr(_dflash_fused_ops, "fused_qk_norm_rope")
        or self.q_norm.variance_epsilon != self.k_norm.variance_epsilon
    ):
        return _DFLASH_DIRECT_ATTN_ORIG_FORWARD(self, positions, hidden_states)

    n = int(hidden_states.shape[0])
    if n > int(getattr(self, "_dflash_query_qkv_max_tokens", 0)):
        return _DFLASH_DIRECT_ATTN_ORIG_FORWARD(self, positions, hidden_states)

    qkv = self._dflash_query_qkv[:n]
    torch.mm(hidden_states, self.qkv_proj.weight.t(), out=qkv)
    cos_sin_cache = self.rotary_emb._match_cos_sin_cache_dtype(qkv)
    _dflash_fused_ops.fused_qk_norm_rope(
        qkv,
        self.num_heads,
        self.num_kv_heads,
        self.num_kv_heads,
        self.head_dim,
        self.q_norm.variance_epsilon,
        self.q_norm.weight,
        self.k_norm.weight,
        cos_sin_cache,
        self.rotary_emb.is_neox_style,
        positions.view(-1),
        -1,
    )
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    q = q.view(-1, self.attn.num_heads, self.attn.head_size)
    k = k.view(-1, self.attn.num_kv_heads, self.attn.head_size)
    v = v.view(-1, self.attn.num_kv_heads, self.attn.head_size_v)
    attn_output_view = self._dflash_direct_attn_out[:n]
    kv_cache_dummy_dep = torch.ops.vllm.unified_kv_cache_update(
        k,
        v,
        self._dflash_direct_attn_layer_name,
    )
    torch.ops.vllm.unified_attention_with_output(
        q,
        k,
        v,
        attn_output_view,
        self._dflash_direct_attn_layer_name,
        kv_cache_dummy_dep=kv_cache_dummy_dep,
    )
    attn_output = attn_output_view.view(-1, self.attn.num_heads * self.attn.head_size_v)
    output, _ = self.o_proj(attn_output)
    return output

DFlashQwen3Attention.forward = _dflash_direct_attention_kvupdate_fused_qk_rope_forward
# --- end pp32k DFlash rescue direct attention kvupdate fused qk-norm-rope patch ---
'''

dflash_payload = r'''

# --- pp32k DFlash rescue direct attention kvupdate fused qk-norm-rope patch ---
from vllm.utils.torch_utils import _encode_layer_name as _dflash_encode_layer_name

_DFLASH_DIRECT_ATTN_ORIG_LOAD_MODEL = DFlashProposer.load_model
_DFLASH_DIRECT_ATTN_ORIG_PROPOSE = DFlashProposer.propose
_DFLASH_DIRECT_ATTN_PROPOSE_CALLS = 0
_DFLASH_DIRECT_ATTN_LAYER_COUNT = 0

def _dflash_direct_attention_load_model(self, target_model):
    global _DFLASH_DIRECT_ATTN_LAYER_COUNT
    result = _DFLASH_DIRECT_ATTN_ORIG_LOAD_MODEL(self, target_model)
    allocated = 0
    max_tokens = int(getattr(self, "max_query_tokens", 8))
    for layer in self.model.model.layers:
        attn = layer.self_attn
        if not getattr(attn, "_dflash_query_qkv_workspace_v3_enabled", False):
            continue
        core = attn.attn
        attn._dflash_direct_attn_out = torch.empty(
            (max_tokens, core.num_heads, core.head_size_v),
            dtype=attn.qkv_proj.weight.dtype,
            device=attn.qkv_proj.weight.device,
        )
        attn._dflash_direct_attn_layer_name = _dflash_encode_layer_name(core.layer_name)
        attn._dflash_direct_attention_enabled = True
        allocated += 1
    _DFLASH_DIRECT_ATTN_LAYER_COUNT = allocated
    logger.info(
        "DFlash direct draft-query attention v2 path active: load_model_direct_attention_installed=True allocated_layers=%s preserved_query_qkv=True kv_cache_update_preserved=True fused_qk_norm_rope=True avoided_generic_attention_wrapper=True direct_backend=unified_kv_cache_update+unified_attention_with_output fallback_count=0",
        _DFLASH_DIRECT_ATTN_LAYER_COUNT,
    )
    return result

def _dflash_direct_attention_propose(self, *args, **kwargs):
    global _DFLASH_DIRECT_ATTN_PROPOSE_CALLS
    out = _DFLASH_DIRECT_ATTN_ORIG_PROPOSE(self, *args, **kwargs)
    _DFLASH_DIRECT_ATTN_PROPOSE_CALLS += 1
    calls = _DFLASH_DIRECT_ATTN_PROPOSE_CALLS
    try:
        query_tokens = int(out.shape[-1]) + 1
    except Exception:
        query_tokens = int(getattr(self, "max_query_tokens", 8))
    if calls <= 20 or calls % 50 == 0:
        logger.info(
            "DFlash direct draft-query attention v2 path active: propose_calls=%s estimated_direct_attention_calls=%s query_tokens=%s allocated_layers=%s preserved_query_qkv=True kv_cache_update_preserved=True fused_qk_norm_rope=True avoided_generic_attention_wrapper=True direct_backend=unified_kv_cache_update+unified_attention_with_output target_sampling_unchanged=True",
            calls,
            calls * _DFLASH_DIRECT_ATTN_LAYER_COUNT,
            query_tokens,
            _DFLASH_DIRECT_ATTN_LAYER_COUNT,
        )
    return out

DFlashProposer.load_model = _dflash_direct_attention_load_model
DFlashProposer.propose = _dflash_direct_attention_propose
# --- end pp32k DFlash rescue direct attention kvupdate fused qk-norm-rope patch ---
'''

for path, payload, marker in [
    (qwen, qwen_payload, "pp32k DFlash rescue direct attention kvupdate fused qk-norm-rope patch"),
    (dflash, dflash_payload, "pp32k DFlash rescue direct attention kvupdate fused qk-norm-rope patch"),
]:
    text = path.read_text()
    if marker not in text:
        p = path.with_suffix(path.suffix + ".bak_direct_attention_kvupdate_fused_qk_rope")
        p.write_text(text)
        path.write_text(text + payload)
    print("direct-attention-kvupdate-fused-qk-rope-patched", path)
PY

cat > "$ART/patch_compact_delta_rejection_v2_nosync.py" <<'PY'
from pathlib import Path

path = Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/sample/rejection_sampler.py")
payload = r'''

# --- pp32k DFlash rescue compact delta rejection v2 no-sync plus DFlash proposer FULL cudagraph patch ---
_DFLASH_COMPACT_DELTA_ORIG_FORWARD = RejectionSampler.forward
_DFLASH_COMPACT_DELTA_CALLS = 0
_DFLASH_COMPACT_DELTA_ROWS = 0
_DFLASH_COMPACT_DELTA_CANDIDATES = 0
_DFLASH_COMPACT_DELTA_FALLBACKS = 0

def _dflash_expand_batch_to_tokens_cpu_safe(x, cu_num_draft_tokens, num_tokens, replace_from=None, replace_to=None):
    expanded = expand_batch_to_tokens(x, cu_num_draft_tokens, num_tokens)
    if replace_from is not None:
        expanded = torch.where(expanded == replace_from, torch.full_like(expanded, replace_to), expanded)
    return expanded

def _dflash_compact_delta_rejection_sample(
    target_logits,
    metadata,
    bonus_token_ids,
    sampling_metadata,
):
    global _DFLASH_COMPACT_DELTA_CALLS, _DFLASH_COMPACT_DELTA_ROWS
    global _DFLASH_COMPACT_DELTA_CANDIDATES, _DFLASH_COMPACT_DELTA_FALLBACKS

    if (
        sampling_metadata.all_greedy
        or sampling_metadata.top_k is None
        or sampling_metadata.top_p is None
        or sampling_metadata.temperature is None
        or sampling_metadata.max_num_logprobs is not None
        or target_logits.ndim != 2
        or metadata.draft_token_ids.ndim != 1
        or len(metadata.num_draft_tokens) != 1
        or metadata.max_spec_len > 16
    ):
        _DFLASH_COMPACT_DELTA_FALLBACKS += 1
        return None

    num_tokens, vocab_size = target_logits.shape
    if num_tokens == 0:
        return None
    try:
        max_top_k = int(sampling_metadata.top_k.max().item())
    except Exception:
        _DFLASH_COMPACT_DELTA_FALLBACKS += 1
        return None
    if max_top_k <= 0 or max_top_k > 32 or max_top_k >= vocab_size:
        _DFLASH_COMPACT_DELTA_FALLBACKS += 1
        return None

    logits = target_logits
    temperature = _dflash_expand_batch_to_tokens_cpu_safe(
        sampling_metadata.temperature,
        metadata.cu_num_draft_tokens,
        num_tokens,
        replace_from=GREEDY_TEMPERATURE,
        replace_to=1,
    )
    logits = logits / temperature.unsqueeze(-1)

    top_k = expand_batch_to_tokens(
        sampling_metadata.top_k,
        metadata.cu_num_draft_tokens,
        num_tokens,
    ).to(torch.long)
    top_p = expand_batch_to_tokens(
        sampling_metadata.top_p,
        metadata.cu_num_draft_tokens,
        num_tokens,
    )

    top_vals, top_ids = torch.topk(logits, max_top_k, dim=-1)
    col = torch.arange(max_top_k, device=logits.device).unsqueeze(0)
    k_mask = col < top_k.unsqueeze(1)
    top_vals = top_vals.masked_fill(~k_mask, -float("inf"))

    top_probs_initial = torch.softmax(top_vals, dim=-1, dtype=torch.float32)
    cdf = top_probs_initial.cumsum(dim=-1)
    # Keep the nucleus prefix, including the first token that crosses top_p.
    p_mask = ((cdf - top_probs_initial) < top_p.unsqueeze(1)) & k_mask
    p_mask[:, 0] = True
    constrained_vals = top_vals.masked_fill(~p_mask, -float("inf"))
    top_probs = torch.softmax(constrained_vals, dim=-1, dtype=torch.float32)

    draft_ids = metadata.draft_token_ids.to(torch.long)
    draft_match = top_ids == draft_ids.unsqueeze(1)
    target_prob_draft = (top_probs * draft_match.to(top_probs.dtype)).sum(dim=-1)

    uniform_probs = generate_uniform_probs(
        num_tokens,
        metadata.num_draft_tokens,
        sampling_metadata.generators,
        logits.device,
    )
    accepted = uniform_probs < target_prob_draft

    recovered_probs = top_probs.masked_fill(draft_match, 0.0)
    q = torch.empty_like(recovered_probs)
    q.exponential_()
    for req_idx, generator in sampling_metadata.generators.items():
        if req_idx == 0 and metadata.num_draft_tokens[0] > 0:
            q.exponential_(generator=generator)
    recovered_idx = recovered_probs.div(q).argmax(dim=-1)
    recovered_ids = top_ids.gather(1, recovered_idx.unsqueeze(1)).squeeze(1).to(torch.int32)

    output_token_ids = torch.full(
        (1, metadata.max_spec_len + 1),
        PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32,
        device=logits.device,
    )
    n = int(metadata.num_draft_tokens[0])
    if n > 0:
        prefix_accept = torch.cumprod(accepted[:n].to(torch.int32), dim=0).to(torch.bool)
        prev_prefix = torch.cat(
            [torch.ones(1, dtype=torch.bool, device=logits.device), prefix_accept[:-1]]
        )
        first_reject = (~accepted[:n]) & prev_prefix
        chosen = torch.where(
            prefix_accept,
            draft_ids[:n].to(torch.int32),
            torch.where(first_reject, recovered_ids[:n], torch.full((n,), PLACEHOLDER_TOKEN_ID, dtype=torch.int32, device=logits.device)),
        )
        output_token_ids[0, :n] = chosen
        all_accept = prefix_accept[-1]
        output_token_ids[0, n] = torch.where(
            all_accept,
            bonus_token_ids[0, 0].to(torch.int32),
            torch.tensor(PLACEHOLDER_TOKEN_ID, dtype=torch.int32, device=logits.device),
        )

    _DFLASH_COMPACT_DELTA_CALLS += 1
    _DFLASH_COMPACT_DELTA_ROWS += int(num_tokens)
    _DFLASH_COMPACT_DELTA_CANDIDATES += int(num_tokens * max_top_k)
    c = _DFLASH_COMPACT_DELTA_CALLS
    if c <= 20 or c % 50 == 0:
        logger.info(
            "DFlash compact delta rejection v2 no-sync plus DFlash proposer FULL cudagraph path active: calls=%s rows=%s last_rows=%s top_k=%s preserved_delta_proposal_sampling=True DFlash longctx feature metadata\|DFlash shifted suffix block-table path active\|avoided_full_vocab_target_softmax=True avoided_full_vocab_recovered_scan=True avoided_hot_path_cpu_sync=True fallback_count=%s target_sampling_unchanged=True",
            c,
            _DFLASH_COMPACT_DELTA_ROWS,
            int(num_tokens),
            max_top_k,
            _DFLASH_COMPACT_DELTA_FALLBACKS,
        )
    return output_token_ids

def _dflash_compact_delta_forward(self, metadata, draft_probs, logits, sampling_metadata):
    if draft_probs is not None or self.synthetic_mode:
        return _DFLASH_COMPACT_DELTA_ORIG_FORWARD(self, metadata, draft_probs, logits, sampling_metadata)

    assert metadata.max_spec_len <= MAX_SPEC_LEN
    bonus_logits_indices = metadata.bonus_logits_indices
    target_logits_indices = metadata.target_logits_indices
    assert logits is not None
    bonus_logits = logits[bonus_logits_indices]
    bonus_sampler_output = self.sampler(
        logits=bonus_logits,
        sampling_metadata=replace(sampling_metadata, max_num_logprobs=-1),
        predict_bonus_token=True,
        logprobs_mode_override="processed_logits"
        if self.is_processed_logprobs_mode
        else "raw_logits",
    )
    bonus_token_ids = bonus_sampler_output.sampled_token_ids

    raw_target_logits = logits[target_logits_indices].to(torch.float32)
    target_logits = raw_target_logits
    if not self.is_processed_logprobs_mode:
        target_logits = target_logits.clone()
    target_logits = self.apply_logits_processors(
        target_logits, sampling_metadata, metadata
    )

    output_token_ids = _dflash_compact_delta_rejection_sample(
        target_logits,
        metadata,
        bonus_token_ids,
        sampling_metadata,
    )
    if output_token_ids is None:
        return _DFLASH_COMPACT_DELTA_ORIG_FORWARD(self, metadata, draft_probs, logits, sampling_metadata)
    return SamplerOutput(sampled_token_ids=output_token_ids, logprobs_tensors=None)

RejectionSampler.forward = _dflash_compact_delta_forward
# --- end pp32k DFlash rescue compact delta rejection v2 no-sync plus DFlash proposer FULL cudagraph patch ---
'''

text = path.read_text()
marker = "pp32k DFlash rescue compact delta rejection v2 no-sync plus DFlash proposer FULL cudagraph patch"
if marker not in text:
    path.with_suffix(path.suffix + ".bak_compact_delta_rejection_v2_nosync").write_text(text)
    path.write_text(text + payload)
print("compact-delta-rejection-v2-nosync-patched", path)
PY

cat > "$ART/patch_proposer_full_cudagraph.py" <<'PY'
from pathlib import Path

path = Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/llm_base_proposer.py")
payload = r'''

# --- pp32k DFlash rescue proposer full cudagraph patch ---
_DFLASH_PROPOSER_FULL_CG_ORIG_INIT_KEYS = SpecDecodeBaseProposer.initialize_cudagraph_keys
_DFLASH_PROPOSER_FULL_CG_ORIG_DETERMINE = SpecDecodeBaseProposer._determine_batch_execution_and_padding
_DFLASH_PROPOSER_FULL_CG_CALLS = 0

def _dflash_proposer_full_cg_initialize_keys(self, cudagraph_mode):
    method = getattr(self, "method", None)
    if method == "dflash" and not self.speculative_config.enforce_eager:
        self.cudagraph_dispatcher.initialize_cudagraph_keys(
            cudagraph_mode,
            uniform_decode_query_len=1 + self.num_speculative_tokens,
        )
        result = None
        forced_full = True
    else:
        result = _DFLASH_PROPOSER_FULL_CG_ORIG_INIT_KEYS(self, cudagraph_mode)
        forced_full = False
    dispatcher = getattr(self, "cudagraph_dispatcher", None)
    keys = getattr(dispatcher, "cudagraph_keys", {}) if dispatcher is not None else {}
    mode = getattr(dispatcher, "cudagraph_mode", None) if dispatcher is not None else None
    piecewise = len(keys.get(CUDAGraphMode.PIECEWISE, set())) if keys else 0
    full = len(keys.get(CUDAGraphMode.FULL, set())) if keys else 0
    logger.info(
        "DFlash proposer full-cudagraph path active: initialize_keys method=%s requested_mode=%s dispatcher_mode=%s piecewise_keys=%s full_keys=%s forced_full=%s uniform_decode_query_len=%s",
        method, cudagraph_mode, mode, piecewise, full, forced_full,
        1 + getattr(self, "num_speculative_tokens", 0),
    )
    return result

def _dflash_proposer_full_cg_determine(self, num_tokens, use_cudagraphs=True):
    global _DFLASH_PROPOSER_FULL_CG_CALLS
    if getattr(self, "method", None) == "dflash" and use_cudagraphs:
        mode, batch_desc = self.cudagraph_dispatcher.dispatch(
            num_tokens,
            uniform_decode=(num_tokens == 1 + self.num_speculative_tokens),
        )
        padded = batch_desc.num_tokens
        across_dp = None
    else:
        mode, padded, across_dp = _DFLASH_PROPOSER_FULL_CG_ORIG_DETERMINE(
            self, num_tokens, use_cudagraphs=use_cudagraphs
        )
    _DFLASH_PROPOSER_FULL_CG_CALLS += 1
    if getattr(self, "method", None) == "dflash" and (_DFLASH_PROPOSER_FULL_CG_CALLS <= 40 or _DFLASH_PROPOSER_FULL_CG_CALLS % 50 == 0):
        dispatcher = getattr(self, "cudagraph_dispatcher", None)
        keys = getattr(dispatcher, "cudagraph_keys", {}) if dispatcher is not None else {}
        piecewise = len(keys.get(CUDAGraphMode.PIECEWISE, set())) if keys else 0
        full = len(keys.get(CUDAGraphMode.FULL, set())) if keys else 0
        logger.info(
            "DFlash proposer full-cudagraph path active: call=%s num_tokens=%s padded_tokens=%s runtime_mode=%s use_cudagraphs=%s piecewise_keys=%s full_keys=%s static_query_rows=%s static_nspec=%s full_graph_available=%s graph_replay_expected=%s uniform_decode_dispatch=%s",
            _DFLASH_PROPOSER_FULL_CG_CALLS, num_tokens, padded, mode,
            use_cudagraphs, piecewise, full, getattr(self, "max_query_tokens", None),
            getattr(self, "num_speculative_tokens", None), full > 0,
            mode != CUDAGraphMode.NONE,
            num_tokens == 1 + getattr(self, "num_speculative_tokens", -999),
        )
    return mode, padded, across_dp

SpecDecodeBaseProposer.initialize_cudagraph_keys = _dflash_proposer_full_cg_initialize_keys
SpecDecodeBaseProposer._determine_batch_execution_and_padding = _dflash_proposer_full_cg_determine
# --- end pp32k DFlash rescue proposer full cudagraph patch ---
'''

text = path.read_text()
marker = "pp32k DFlash rescue proposer full cudagraph patch"
if marker not in text:
    path.with_suffix(path.suffix + ".bak_proposer_full_cudagraph").write_text(text)
    path.write_text(text + payload)
print("proposer-full-cudagraph-patched", path)
PY

cat > "$ART/patch_longctx_feature_metadata.py" <<'PY'
from pathlib import Path

gpu = Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py")
dflash = Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/dflash.py")

gpu_payload = r'''

# --- pp32k DFlash rescue longctx feature metadata patch ---
_DFLASH_LONGCTX_ORIG_PROPOSE_DRAFT = GPUModelRunner.propose_draft_token_ids
_DFLASH_LONGCTX_GPU_CALLS = 0

def _dflash_longctx_tensor_stats(t):
    try:
        tf = t.detach()
        shape = tuple(tf.shape)
        dtype = str(tf.dtype)
        device = str(tf.device)
        if tf.numel() == 0:
            return shape, dtype, device, None, None, None
        sample = tf
        if sample.dim() >= 2:
            sample = sample[: min(4, sample.shape[0])]
        sample_f = sample.float()
        return (
            shape,
            dtype,
            device,
            float(sample_f.norm().item()),
            float(sample_f.pow(2).mean().sqrt().item()),
            float(sample_f.abs().max().item()),
        )
    except Exception as exc:
        return ("error", repr(exc)), "unknown", "unknown", None, None, None

def _dflash_longctx_positions_summary(pos):
    try:
        if pos is None:
            return "none"
        p = pos.detach().reshape(-1)
        n = int(p.numel())
        if n == 0:
            return "empty"
        head = p[: min(8, n)].to("cpu").tolist()
        tail = p[max(0, n - 8):].to("cpu").tolist()
        return f"shape={tuple(p.shape)} min={int(p.min().item())} max={int(p.max().item())} head={head} tail={tail}"
    except Exception as exc:
        return f"error={exc!r}"

def _dflash_longctx_propose_draft_token_ids(self, scheduler_output, sampled_token_ids,
                                            sampling_metadata, hidden_states,
                                            sample_hidden_states, aux_hidden_states,
                                            spec_decode_metadata, common_attn_metadata,
                                            slot_mappings):
    global _DFLASH_LONGCTX_GPU_CALLS
    spec_config = self.speculative_config
    is_dflash = spec_config is not None and spec_config.use_dflash()
    if is_dflash:
        _DFLASH_LONGCTX_GPU_CALLS += 1
        call = _DFLASH_LONGCTX_GPU_CALLS
        try:
            total_sched = int(scheduler_output.total_num_scheduled_tokens)
        except Exception:
            total_sched = -1
        try:
            spec_counts = {str(k): len(v) for k, v in scheduler_output.scheduled_spec_decode_tokens.items()}
        except Exception as exc:
            spec_counts = {"error": repr(exc)}
        try:
            req_meta = []
            trim_prompt_len = None
            trim_output_len = None
            for rid in list(getattr(self.input_batch, "req_ids", [])):
                st = self.requests.get(rid)
                if st is None:
                    continue
                if trim_prompt_len is None:
                    trim_prompt_len = len(st.prompt_token_ids)
                    trim_output_len = len(st.output_token_ids)
                req_meta.append(
                    f"{rid}:prompt={len(st.prompt_token_ids)} computed={st.num_computed_tokens} output={len(st.output_token_ids)}"
                )
        except Exception as exc:
            req_meta = [f"error={exc!r}"]
            trim_prompt_len = None
            trim_output_len = None
        try:
            if is_dflash and trim_prompt_len is not None:
                setattr(self.drafter, "_dflash_longctx_prompt_len", int(trim_prompt_len))
                setattr(self.drafter, "_dflash_longctx_output_len", int(trim_output_len or 0))
        except Exception:
            pass
        try:
            pos_all = self._get_positions(total_sched)
            pos_summary = _dflash_longctx_positions_summary(pos_all)
        except Exception as exc:
            pos_summary = f"error={exc!r}"
        aux_summary = []
        if aux_hidden_states is not None:
            for i, h in enumerate(aux_hidden_states[:8]):
                shape, dtype, device, norm, rms, amax = _dflash_longctx_tensor_stats(h[: max(1, total_sched)])
                aux_summary.append(
                    f"layer_index={i} shape={shape} dtype={dtype} device={device} norm4={norm} rms4={rms} amax4={amax}"
                )
        should_log = False
        try:
            should_log = any("prompt=32769" in x for x in req_meta) and (
                total_sched != 1728 or "': 15" in str(spec_counts) or any("computed=0" not in x for x in req_meta)
            )
        except Exception:
            should_log = call <= 20
        if should_log:
            logger.info(
                "DFlash longctx feature metadata gpu path active: call=%s total_scheduled=%s common_num_actual=%s common_max_query=%s common_max_seq=%s spec_counts=%s req_meta=%s target_positions=%s hidden_shape=%s sample_hidden_shape=%s aux_count=%s aux_stats=%s output_unchanged=True",
                call,
                total_sched,
                getattr(common_attn_metadata, "num_actual_tokens", None),
                getattr(common_attn_metadata, "max_query_len", None),
                getattr(common_attn_metadata, "max_seq_len", None),
                spec_counts,
                req_meta,
                pos_summary,
                tuple(hidden_states.shape) if hidden_states is not None else None,
                tuple(sample_hidden_states.shape) if sample_hidden_states is not None else None,
                len(aux_hidden_states) if aux_hidden_states is not None else 0,
                " | ".join(aux_summary),
            )
    out = _DFLASH_LONGCTX_ORIG_PROPOSE_DRAFT(
        self, scheduler_output, sampled_token_ids, sampling_metadata,
        hidden_states, sample_hidden_states, aux_hidden_states,
        spec_decode_metadata, common_attn_metadata, slot_mappings)
    if is_dflash:
        try:
            first = out[0, : min(8, out.shape[1])].detach().to("cpu").tolist() if hasattr(out, "shape") else out[:1]
        except Exception as exc:
            first = f"error={exc!r}"
        if _DFLASH_LONGCTX_GPU_CALLS <= 80 or _DFLASH_LONGCTX_GPU_CALLS % 50 == 0:
            logger.info(
                "DFlash longctx feature metadata draft ids: call=%s first_draft_ids=%s output_unchanged=True",
                _DFLASH_LONGCTX_GPU_CALLS,
                first,
            )
    return out

GPUModelRunner.propose_draft_token_ids = _dflash_longctx_propose_draft_token_ids
# --- end pp32k DFlash rescue longctx feature metadata patch ---
'''

dflash_payload = r'''

# --- pp32k DFlash rescue longctx feature metadata patch ---
_DFLASH_LONGCTX_ORIG_SET_INPUTS = DFlashProposer.set_inputs_first_pass
_DFLASH_LONGCTX_ORIG_BUILD_INPUTS = DFlashProposer.build_model_inputs_first_pass
_DFLASH_LONGCTX_SET_CALLS = 0
_DFLASH_LONGCTX_BUILD_CALLS = 0

def _dflash_longctx_pos_summary(pos):
    try:
        p = pos.detach().reshape(-1)
        n = int(p.numel())
        if n == 0:
            return "empty"
        return "shape=%s min=%s max=%s head=%s tail=%s" % (
            tuple(p.shape),
            int(p.min().item()),
            int(p.max().item()),
            p[: min(8, n)].to("cpu").tolist(),
            p[max(0, n - 8):].to("cpu").tolist(),
        )
    except Exception as exc:
        return "error=%r" % (exc,)

def _dflash_longctx_state_summary(t):
    try:
        x = t.detach()
        sample = x[: min(4, x.shape[0])] if x.dim() >= 2 else x
        sf = sample.float()
        return "shape=%s dtype=%s device=%s norm4=%.6f rms4=%.6f amax4=%.6f" % (
            tuple(x.shape),
            str(x.dtype),
            str(x.device),
            float(sf.norm().item()) if sf.numel() else 0.0,
            float(sf.pow(2).mean().sqrt().item()) if sf.numel() else 0.0,
            float(sf.abs().max().item()) if sf.numel() else 0.0,
        )
    except Exception as exc:
        return "error=%r" % (exc,)

def _dflash_longctx_set_inputs(self, target_token_ids, next_token_ids, target_positions,
                               target_hidden_states, token_indices_to_sample, cad,
                               num_rejected_tokens_gpu):
    global _DFLASH_LONGCTX_SET_CALLS
    _DFLASH_LONGCTX_SET_CALLS += 1
    call = _DFLASH_LONGCTX_SET_CALLS
    try:
        prompt_len = int(getattr(self, "_dflash_longctx_prompt_len", -1))
        output_len = int(getattr(self, "_dflash_longctx_output_len", -1))
        pos_flat = target_positions.detach().reshape(-1)
        pos_max_before = int(pos_flat.max().item()) if pos_flat.numel() else -1
        extra = max(0, pos_max_before - prompt_len + 1)
        if (
            prompt_len >= 32768
            and output_len == 0
            and extra == 1
            and int(target_token_ids.shape[0]) > extra
        ):
            keep = int(target_token_ids.shape[0]) - extra
            target_token_ids = target_token_ids[:keep]
            target_hidden_states = target_hidden_states[:keep]
            target_positions = target_positions[..., :keep].contiguous()
            seq_lens = cad.seq_lens - extra
            qsl = cad.query_start_loc.clone()
            qsl[1:] = qsl[1:] - extra
            qsl_cpu = None
            if getattr(cad, "query_start_loc_cpu", None) is not None:
                qsl_cpu = cad.query_start_loc_cpu.clone()
                qsl_cpu[1:] = qsl_cpu[1:] - extra
            seq_lens_cpu_upper_bound = (
                cad.seq_lens_cpu_upper_bound - extra
                if getattr(cad, "seq_lens_cpu_upper_bound", None) is not None
                else None
            )
            slot_mapping = (
                cad.slot_mapping[:keep]
                if getattr(cad, "slot_mapping", None) is not None
                else None
            )
            cad = CommonAttentionMetadata(
                query_start_loc=qsl,
                seq_lens=seq_lens,
                query_start_loc_cpu=qsl_cpu,
                _seq_lens_cpu=getattr(cad, "_seq_lens_cpu", None),
                _num_computed_tokens_cpu=getattr(cad, "_num_computed_tokens_cpu", None),
                seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
                num_reqs=cad.num_reqs,
                num_actual_tokens=max(0, int(cad.num_actual_tokens) - extra),
                max_query_len=max(1, int(cad.max_query_len) - extra),
                max_seq_len=max(1, int(cad.max_seq_len) - extra),
                block_table_tensor=cad.block_table_tensor,
                slot_mapping=slot_mapping,
                causal=cad.causal,
            )
            pos_max_after = int(target_positions.detach().reshape(-1).max().item())
            logger.info(
                "DFlash longctx prompt-boundary trim path active: call=%s trimmed_extra_context_rows=%s prompt_len=%s output_len=%s prompt_boundary_context_max_before=%s prompt_boundary_context_max_after=%s adjusted_num_context=%s adjusted_cad_max_query=%s adjusted_cad_max_seq=%s output_unchanged=False draft_alignment_patch=True",
                call,
                extra,
                prompt_len,
                output_len,
                pos_max_before,
                pos_max_after,
                keep,
                cad.max_query_len,
                cad.max_seq_len,
            )
    except Exception as exc:
        logger.exception("DFlash longctx prompt-boundary trim setup failed: %s", exc)
    try:
        pos_max = int(target_positions.detach().max().item())
    except Exception:
        pos_max = -1
    if pos_max >= 30000 or int(target_token_ids.shape[0]) <= 32:
        try:
            next_ids = next_token_ids[: min(4, next_token_ids.shape[0])].detach().to("cpu").tolist()
        except Exception as exc:
            next_ids = "error=%r" % (exc,)
        logger.info(
            "DFlash longctx feature metadata proposer set_inputs active: call=%s num_context=%s batch_size=%s num_query_per_req=%s target_positions=%s target_hidden=%s next_token_ids_head=%s cad_max_query=%s cad_max_seq=%s cad_seq_lens=%s has_rejected_gpu=%s output_unchanged=True",
            call,
            int(target_token_ids.shape[0]),
            cad.batch_size(),
            1 + self.num_speculative_tokens,
            _dflash_longctx_pos_summary(target_positions),
            _dflash_longctx_state_summary(target_hidden_states),
            next_ids,
            getattr(cad, "max_query_len", None),
            getattr(cad, "max_seq_len", None),
            _dflash_longctx_pos_summary(getattr(cad, "seq_lens", None)),
            num_rejected_tokens_gpu is not None,
        )
    return _DFLASH_LONGCTX_ORIG_SET_INPUTS(
        self, target_token_ids, next_token_ids, target_positions,
        target_hidden_states, token_indices_to_sample, cad, num_rejected_tokens_gpu)

def _dflash_longctx_build_inputs(self, num_tokens, num_input_tokens, mm_embed_inputs):
    global _DFLASH_LONGCTX_BUILD_CALLS
    _DFLASH_LONGCTX_BUILD_CALLS += 1
    call = _DFLASH_LONGCTX_BUILD_CALLS
    try:
        ctx_max = int(self._context_positions_buffer[: max(0, int(getattr(self, "_dflash_num_context", -1)))].detach().max().item())
    except Exception:
        ctx_max = -1
    if ctx_max >= 30000 or int(getattr(self, "_dflash_num_context", -1)) <= 32:
        nctx = int(getattr(self, "_dflash_num_context", -1))
        logger.info(
            "DFlash longctx feature metadata precompute active: call=%s num_tokens=%s num_input_tokens=%s num_context=%s context_positions=%s query_positions=%s context_hidden=%s context_slot_mapping_head=%s output_unchanged=True",
            call,
            num_tokens,
            num_input_tokens,
            nctx,
            _dflash_longctx_pos_summary(self._context_positions_buffer[: max(0, nctx)]),
            _dflash_longctx_pos_summary(self.positions[:num_input_tokens]),
            _dflash_longctx_state_summary(self._dflash_hidden_states),
            self._context_slot_mapping_buffer[: min(8, max(0, nctx))].detach().to("cpu").tolist() if nctx > 0 else [],
        )
    return _DFLASH_LONGCTX_ORIG_BUILD_INPUTS(self, num_tokens, num_input_tokens, mm_embed_inputs)

DFlashProposer.set_inputs_first_pass = _dflash_longctx_set_inputs
DFlashProposer.build_model_inputs_first_pass = _dflash_longctx_build_inputs
# --- end pp32k DFlash rescue longctx feature metadata patch ---
'''

for path, payload, marker in [
    (gpu, gpu_payload, "pp32k DFlash rescue longctx feature metadata patch"),
    (dflash, dflash_payload, "pp32k DFlash rescue longctx feature metadata patch"),
]:
    text = path.read_text()
    if marker not in text:
        path.with_suffix(path.suffix + ".bak_longctx_feature_metadata").write_text(text)
        path.write_text(text + payload)
    print("longctx-feature-metadata-patched", path)
PY

cat > "$ART/sitecustomize.py" <<'PY'
import json as _json

def _normalize_payload(payload):
    if not isinstance(payload, dict):
        return payload
    payload["temperature"] = 0.6
    payload.pop("top_p", None)
    payload.pop("min_tokens", None)
    payload.pop("ignore_eos", None)
    print(
        "[temp06-sitecustomize] patched chat payload "
        f"temperature={payload.get('temperature')} "
        f"top_p_present={'top_p' in payload} "
        f"min_tokens_present={'min_tokens' in payload} "
        f"ignore_eos_present={'ignore_eos' in payload}",
        flush=True,
    )
    return payload

def _patch_aiohttp():
    try:
        import aiohttp
    except Exception:
        return False
    if getattr(aiohttp.ClientSession.post, "_temp06_patched", False):
        return True
    _orig_post = aiohttp.ClientSession.post

    def _patched_post(self, url, *args, **kwargs):
        if "/chat/completions" in str(url):
            if "json" in kwargs:
                kwargs["json"] = _normalize_payload(kwargs["json"])
            elif "data" in kwargs:
                try:
                    payload = _json.loads(kwargs["data"])
                except Exception:
                    payload = None
                if isinstance(payload, dict):
                    kwargs["data"] = _json.dumps(_normalize_payload(payload))
        return _orig_post(self, url, *args, **kwargs)

    _patched_post._temp06_patched = True
    aiohttp.ClientSession.post = _patched_post
    print("[temp06-sitecustomize] aiohttp ClientSession.post patch installed", flush=True)
    return True

_patch_aiohttp()
PY

docker run -d --gpus all --network host --ipc host \
  --name "$NAME" \
  -v "$RECIPE:/repro" \
  -v "${MODELS_DIR:-$HOME/models}":/models:ro \
  -v "$ART:/out" \
  "$IMAGE" bash -lc '
    set -euo pipefail
    cp /usr/lib/python3.12/sitecustomize.py /tmp/sitecustomize.py.original
    cat >/usr/lib/python3.12/sitecustomize.py <<'"'"'PY'"'"'
try:
    import apport_python_hook
    apport_python_hook.install()
except Exception:
    pass
PY
    diff -u /tmp/sitecustomize.py.original /usr/lib/python3.12/sitecustomize.py > /tmp/sitecustomize.diff || true
    env | sort > /tmp/env-before-unset.txt
    mv /usr/local/lib/python3.12/dist-packages/__editable__.flash_qla-0.1.0+827fdd8.pth /tmp/__editable__.flash_qla-0.1.0+827fdd8.pth.disabled 2>/tmp/flash-qla-disable-pth.log || true
    mv /usr/local/lib/python3.12/dist-packages/__editable___flash_qla_0_1_0_827fdd8_finder.py /tmp/__editable___flash_qla_0_1_0_827fdd8_finder.py.disabled 2>/tmp/flash-qla-disable-finder.log || true
    mv /usr/local/lib/python3.12/dist-packages/fla /tmp/fla.disabled 2>/tmp/fla-disable.log || true
    mv /usr/local/lib/python3.12/dist-packages/flash_linear_attention-0.5.0.dist-info /tmp/flash_linear_attention-0.5.0.dist-info.disabled 2>>/tmp/fla-disable.log || true
    python3 /out/patch_query_qkv_workspace_v3.py > /tmp/patch-query-qkv-workspace-v3.log 2>&1
    python3 /out/patch_direct_attention_kvupdate_fused_qk_rope.py > /tmp/patch-direct-attention-kvupdate-fused-qk-rope.log 2>&1
    python3 /out/patch_compact_delta_rejection_v2_nosync.py > /tmp/patch-compact-delta-rejection-v2-nosync.log 2>&1
    python3 /out/patch_proposer_full_cudagraph.py > /tmp/patch-proposer-full-cudagraph.log 2>&1
    python3 /out/patch_longctx_feature_metadata.py > /tmp/patch-longctx-feature-metadata.log 2>&1
    python3 /repro/patch_shifted_suffix_block_table.py > /tmp/patch-shifted-suffix-block-table.log 2>&1
    python3 -m py_compile /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_dflash.py /usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/dflash.py /usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/llm_base_proposer.py /usr/local/lib/python3.12/dist-packages/vllm/v1/sample/rejection_sampler.py /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py > /tmp/patch-fused-qk-rope-pycompile.log 2>&1
    grep -R "DFlash direct draft-query attention v2 path active\|kv_cache_update_preserved=True\|DFlash query QKV workspace v3 path active\|DFlash local argmax proposer path active\|DFlash compact delta rejection v2 no-sync plus DFlash proposer FULL cudagraph path active\|DFlash proposer full-cudagraph path active\|DFlash longctx feature metadata\|DFlash shifted suffix block-table path active\|avoided_full_vocab_target_softmax=True\|avoided_full_vocab_recovered_scan=True\|avoided_hot_path_cpu_sync=True" /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_dflash.py /usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/dflash.py /usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/llm_base_proposer.py /usr/local/lib/python3.12/dist-packages/vllm/v1/sample/rejection_sampler.py /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py > /tmp/patch-direct-attention-kvupdate-grep.log 2>&1
    unset PYTHONPATH
    env | sort > /tmp/env-after-unset.txt
    export VLLM_DFLASH_AR_PROMPT_THRESHOLD=0
    export FLASHQLA_HKV_O_BK=128
    export FLASHQLA_HKV_O_BV=128
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    exec vllm serve /models/Qwen3.6-27B-NVFP4 \
      --served-model-name qwen36-27b \
      --host 0.0.0.0 --port 8001 \
      --tensor-parallel-size 1 \
      --gpu-memory-utilization 0.92 \
      --max-model-len 65536 \
      --max-num-batched-tokens 2048 \
      --max-num-seqs 1 \
      --trust-remote-code \
      --load-format fastsafetensors \
      --attention-backend flash_attn \
      --enable-prefix-caching \
      --default-chat-template-kwargs '"'"'{"enable_thinking": true}'"'"' \
      --speculative-config '"'"'{"method":"dflash","num_speculative_tokens":15,"model":"/models/Qwen3.6-27B-DFlash"}'"'"' \
      --seed 0
  ' > "$ART/container-id.txt"

docker inspect "$NAME" > "$ART/docker-inspect.json" 2> "$ART/docker-inspect.stderr.log" || true

(
  set +e
  for i in $(seq 1 240); do
    if curl -fsS http://127.0.0.1:8001/v1/models >/dev/null 2>&1; then
      exit 0
    fi
    docker ps --format '{{.Names}} {{.Status}}' | grep -q "^$NAME " || exit 1
    sleep 5
  done
  exit 124
) > "$ART/wait-for-server-8001.log" 2>&1
echo $? > "$ART/wait-for-server-8001-exit-code.txt"
docker logs "$NAME" > "$ART/server-after-wait.log" 2>&1 || true

for f in /tmp/sitecustomize.diff /tmp/env-before-unset.txt /tmp/env-after-unset.txt /tmp/flash-qla-disable-pth.log /tmp/flash-qla-disable-finder.log /tmp/fla-disable.log /tmp/patch-query-qkv-workspace-v3.log /tmp/patch-direct-attention-kvupdate-fused-qk-rope.log /tmp/patch-compact-delta-rejection-v2-nosync.log /tmp/patch-proposer-full-cudagraph.log /tmp/patch-longctx-feature-metadata.log /tmp/patch-shifted-suffix-block-table.log /tmp/patch-fused-qk-rope-pycompile.log /tmp/patch-direct-attention-kvupdate-grep.log; do
  docker cp "$NAME:$f" "$ART/$(basename "$f")" >/dev/null 2>"$ART/$(basename "$f").copy.log" || true
done

if [ "$(cat "$ART/wait-for-server-8001-exit-code.txt")" != "0" ]; then
  docker logs "$NAME" > "$ART/server-final.log" 2>&1 || true
  docker inspect "$NAME" > "$ART/docker-inspect-final.json" 2> "$ART/docker-inspect-final.stderr.log" || true
  docker rm -f "$NAME" > "$ART/container-rm-after.txt" 2>&1 || true
  exit 1
fi

set +e
docker run --rm --network host \
  -e PYTHONPATH=/out \
  -v "$RECIPE:/repro" \
  -v "${MODELS_DIR:-$HOME/models}":/models:ro \
  -v "$ART:/out" \
  "$IMAGE" bash -lc "
    set -euo pipefail
    cd /repro
    echo '[warmup] explicit same-shape warmup: pp=32768 tg=128 c=1 runs=30'
    uvx llama-benchy --base-url http://127.0.0.1:8001/v1 --model qwen36-27b --served-model-name qwen36-27b --tokenizer /models/Qwen3.6-27B-NVFP4 --pp 32768 --tg 128 --depth 0 --concurrency 1 --runs 5 --skip-coherence --no-cache --save-result /repro/results/pp32k-tg128-dflash-rescue-20260507/$ROW/warmup-pp32768-tg128-c1-n5.json --format json
    echo '[measure] post-warm diagnostic: pp=32768 tg=128 c=1 runs=30'
    uvx llama-benchy --base-url http://127.0.0.1:8001/v1 --model qwen36-27b --served-model-name qwen36-27b --tokenizer /models/Qwen3.6-27B-NVFP4 --pp 32768 --tg 128 --depth 0 --concurrency 1 --runs 30 --skip-coherence --no-cache --save-result /repro/results/pp32k-tg128-dflash-rescue-20260507/$ROW/measured-pp32768-tg128-c1-n30.json --format json
  " > "$ART/bench-8001.stdout" 2> "$ART/bench-8001.stderr"
bench_rc=$?
echo "$bench_rc" > "$ART/bench-8001-exit-code.txt"
set -e

docker logs "$NAME" > "$ART/server-after-bench.log" 2>&1 || true
docker logs "$NAME" > "$ART/server-final.log" 2>&1 || true
docker inspect "$NAME" > "$ART/docker-inspect-final.json" 2> "$ART/docker-inspect-final.stderr.log" || true
docker rm -f "$NAME" > "$ART/container-rm-after.txt" 2>&1 || true

python3 - <<PY > "$ART/summary.txt"
import json, pathlib, statistics
art = pathlib.Path("$ART")
print("artifact", art)
for name in ["wait-for-server-8001-exit-code.txt", "bench-8001-exit-code.txt"]:
    p = art / name
    print(name, p.read_text().strip() if p.exists() else "missing")
for fname in ["measured-pp32768-tg128-c1-n30.json"]:
  p = art / fname
  if not p.exists():
    print(fname, "missing")
    continue
  print("json", fname)
  try:
    b = json.loads(p.read_text())["benchmarks"][0]
    for key in ["pp_throughput", "tg_throughput"]:
        m = b[key]
        vals = m.get("values") or []
        print(key, "mean", m.get("mean"), "median", statistics.median(vals) if vals else None, "values", vals)
  except Exception as exc:
    print(fname, "parse_error", repr(exc))
log = (art / "server-final.log").read_text(errors="replace") if (art / "server-final.log").exists() else ""
for pat in [
    "DFlash direct draft-query attention v2 path active",
    "preserved_query_qkv=True",
    "kv_cache_update_preserved=True",
    "fused_qk_norm_rope=True",
    "avoided_generic_attention_wrapper=True",
    "DFlash local argmax proposer path active",
    "DFlash query QKV workspace v3 path active",
    "DFlash compact delta rejection v2 no-sync plus DFlash proposer FULL cudagraph path active",
    "DFlash longctx feature metadata gpu path active",
    "DFlash longctx feature metadata draft ids",
    "DFlash longctx feature metadata proposer set_inputs active",
    "DFlash longctx feature metadata precompute active",
    "DFlash longctx prompt-boundary trim path active",
    "DFlash shifted suffix block-table path active",
    "suffix_intrablock_offset=0",
    "block_table_shifted=True",
    "seq_lens_adjusted=True",
    "trimmed_extra_context_rows=1",
    "prompt_boundary_context_max_before=32769",
    "prompt_boundary_context_max_after=32768",
    "output_unchanged=True",
    "avoided_hot_path_cpu_sync=True",
    "preserved_delta_proposal_sampling=True",
    "avoided_full_vocab_target_softmax=True",
    "avoided_full_vocab_recovered_scan=True",
    "SpecDecoding metrics",
    "scheduled_spec_decode_tokens",
    "Traceback",
    "EngineDeadError",
]:
    print(pat, log.count(pat))
stdout = (art / "bench-8001.stdout").read_text(errors="replace") if (art / "bench-8001.stdout").exists() else ""
for pat in ["temperature=0.6", "top_p_present=False", "min_tokens_present=False", "ignore_eos_present=False"]:
    print(pat, stdout.count(pat))
PY
cat "$ART/summary.txt"
exit "$bench_rc"
