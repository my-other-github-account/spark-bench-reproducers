from __future__ import annotations

import copy
import threading
from typing import Any

import torch

_LOCK = threading.Lock()
_STATS: dict[str, int] = {
    "forward_calls": 0,
    "backward_calls": 0,
    "grouped_experts": 0,
    "max_nodes_per_projection": 0,
    "grad_weight_bmm_launches": 0,
    "reduction_kernel_launches": 0,
}
_TARGET_MODULE: Any | None = None


def reset_layer_graph_vjp() -> None:
    with _LOCK:
        for key in _STATS:
            _STATS[key] = 0


def layer_graph_vjp_stats() -> dict[str, int]:
    with _LOCK:
        return copy.deepcopy(_STATS)


def layer_graph_forward(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    payloads: dict[str, dict[str, torch.Tensor]],
    *,
    limit: float,
) -> torch.Tensor:
    """Vectorize all balanced routed experts into two projection graph nodes."""
    if hidden_states.ndim != 2 or top_k_index.ndim != 2:
        raise ValueError("layer graph expects rank-2 hidden states and routing")
    if tuple(top_k_weights.shape) != tuple(top_k_index.shape):
        raise ValueError("routing index/weight shape drift")
    experts = int(payloads["13"]["dense"].shape[0])
    if int(payloads["2"]["dense"].shape[0]) != experts:
        raise ValueError("projection expert cardinality drift")
    tokens, top_k = map(int, top_k_index.shape)
    counts = torch.bincount(top_k_index.reshape(-1), minlength=experts)
    routes = tokens * top_k
    if routes % experts or not bool(torch.all(counts == routes // experts)):
        raise ValueError("layer graph requires balanced nonempty expert routing")
    routes_per_expert = routes // experts

    # Match the incumbent torch.where(mask[expert]) order: expert-major, then
    # top-k slot, then token. This keeps both forward scatter and gradient sums
    # as close as possible to the accepted per-expert graph.
    slot_major_experts = top_k_index.transpose(0, 1).reshape(-1)
    order = torch.argsort(slot_major_experts, stable=True)
    token = torch.arange(tokens, device=top_k_index.device).repeat(top_k)
    slot = torch.arange(top_k, device=top_k_index.device).repeat_interleave(tokens)
    token = token[order]
    slot = slot[order]
    activations13 = hidden_states.index_select(0, token).reshape(
        experts, routes_per_expert, hidden_states.shape[1]
    )
    row13 = payloads["13"]
    projected13 = LayerProjectionKMajorFn.apply(
        activations13,
        row13["codebook"],
        row13["codes"],
        row13["scales"],
        row13["dense"],
    )
    gate, up = projected13.chunk(2, dim=-1)
    intermediate = torch.nn.functional.silu(gate.clamp(max=limit)) * up.clamp(
        min=-limit, max=limit
    )
    row2 = payloads["2"]
    projected2 = LayerProjectionKMajorFn.apply(
        intermediate,
        row2["codebook"],
        row2["codes"],
        row2["scales"],
        row2["dense"],
    )
    weighted = projected2.reshape(routes, hidden_states.shape[1]) * top_k_weights[
        token, slot, None
    ]
    return torch.zeros_like(hidden_states).index_add(0, token, weighted)


def _eager_grouped_codebook_vjp_from_weights(
    grad_weight: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    codebook_shape: tuple[int, ...],
) -> torch.Tensor:
    experts = int(grad_weight.shape[0])
    code_dim = int(codebook_shape[1])
    scale_columns = torch.exp2(scales.float() - 127.0).repeat_interleave(32, -1)
    grouped = (grad_weight.float() * scale_columns).reshape(
        experts, codes.shape[1], codes.shape[2], code_dim
    )
    partial = torch.zeros(
        experts,
        *codebook_shape,
        dtype=torch.float32,
        device=grad_weight.device,
    )
    for expert in range(experts):
        partial[expert].index_add_(
            0,
            codes[expert].reshape(-1).long(),
            grouped[expert].reshape(-1, code_dim),
        )
    return partial.sum(dim=0)


class LayerProjectionKMajorFn(torch.autograd.Function):
    """One autograd node for every expert in one layer projection."""

    @staticmethod
    def forward(ctx, activations, codebook32, codes, scales, dense_ekn):
        if activations.ndim != 3 or dense_ekn.ndim != 3:
            raise ValueError("layer-projection K-major tensors must be rank 3")
        if int(activations.shape[0]) != int(dense_ekn.shape[0]):
            raise ValueError("expert cardinality drift")
        if int(activations.shape[2]) != int(dense_ekn.shape[1]):
            raise ValueError("activation/tile K dimension drift")
        if codes.requires_grad or scales.requires_grad or dense_ekn.requires_grad:
            raise ValueError("packed planes and detached K-major slab must stay frozen")
        activations = activations.contiguous()
        ctx.save_for_backward(activations, codes, scales, dense_ekn)
        ctx.codebook_shape = tuple(codebook32.shape)
        result = torch.bmm(activations, dense_ekn)
        with _LOCK:
            _STATS["forward_calls"] += 1
            _STATS["grouped_experts"] += int(activations.shape[0])
            _STATS["max_nodes_per_projection"] = max(
                _STATS["max_nodes_per_projection"], 1
            )
        if _TARGET_MODULE is not None:
            _TARGET_MODULE._SENTINEL["bmm_launches"] += 1
        return result

    @staticmethod
    def backward(ctx, grad_out):
        activations, codes, scales, dense_ekn = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        grad_activations = torch.bmm(grad_out, dense_ekn.transpose(1, 2))
        grad_weight = torch.bmm(grad_out.transpose(1, 2), activations)
        with _LOCK:
            _STATS["grad_weight_bmm_launches"] += 1
        if grad_out.is_cuda:
            from .kmajor_fused import fused_grouped_codebook_vjp

            grad_codebook = fused_grouped_codebook_vjp(
                grad_weight,
                codes,
                scales,
                int(ctx.codebook_shape[0]),
                int(ctx.codebook_shape[1]),
            )
            with _LOCK:
                _STATS["reduction_kernel_launches"] += 1
        else:
            grad_codebook = _eager_grouped_codebook_vjp_from_weights(
                grad_weight, codes, scales, ctx.codebook_shape
            )
        with _LOCK:
            _STATS["backward_calls"] += 1
        if _TARGET_MODULE is not None:
            _TARGET_MODULE._SENTINEL["backward_calls"] += 1
        return grad_activations, grad_codebook, None, None, None


def install_layer_graph_vjp(surface_module: Any, kmajor_module: Any) -> dict[str, Any]:
    """Install the layer-level graph path behind GenesisPhysicalExperts.forward."""
    global _TARGET_MODULE
    if getattr(surface_module, "_banana_smasher_layer_graph_installed", False):
        return layer_graph_vjp_stats()
    _TARGET_MODULE = kmajor_module
    original_forward = surface_module.GenesisPhysicalExperts.forward
    original_reset = kmajor_module.reset_kmajor_sentinel
    original_sentinel = kmajor_module.kmajor_sentinel

    def graph_forward(self, hidden_states, top_k_index, top_k_weights):
        payloads = getattr(self, "_banana_smasher_graph_payloads", None)
        if payloads is None or not bool(self.use_kmajor_10x):
            return original_forward(self, hidden_states, top_k_index, top_k_weights)
        return layer_graph_forward(
            hidden_states,
            top_k_index,
            top_k_weights,
            payloads,
            limit=float(self.limit),
        )

    def reset_kmajor_sentinel(*, clear_cache: bool = False) -> None:
        original_reset(clear_cache=clear_cache)
        reset_layer_graph_vjp()

    def kmajor_sentinel() -> dict[str, Any]:
        value = original_sentinel()
        value["layer_graph_vjp"] = layer_graph_vjp_stats()
        return value

    surface_module.GenesisPhysicalExperts.forward = graph_forward
    kmajor_module.reset_kmajor_sentinel = reset_kmajor_sentinel
    kmajor_module.kmajor_sentinel = kmajor_sentinel
    surface_module._banana_smasher_layer_graph_installed = True
    reset_layer_graph_vjp()
    return layer_graph_vjp_stats()


def prepare_layer_graph_vjp(layers: Any, kmajor_module: Any) -> dict[str, Any]:
    """Consolidate the sealed 512 logical tiles into two contiguous slabs."""
    if not layers:
        raise ValueError("layer graph preparation requires at least one layer")
    first = layers[0]
    experts = 256
    shared: dict[str, dict[str, Any]] = {}
    cache_keys: list[tuple[Any, ...]] = []
    dense_values: list[torch.Tensor] = []
    for projection in ("13", "2"):
        rows = first._payloads_for(projection, list(range(experts)))
        if set(rows) != set(range(experts)):
            raise RuntimeError(f"incomplete layer graph payloads for projection {projection}")
        codebook_keys = {str(rows[expert][4]) for expert in range(experts)}
        tiers = {str(rows[expert][1]) for expert in range(experts)}
        if len(codebook_keys) != 1 or None in {rows[expert][4] for expert in range(experts)}:
            raise RuntimeError(
                f"layer graph requires one shared VQ codebook for projection {projection}"
            )
        codebook_key = next(iter(codebook_keys))
        codebook = first.codebooks[codebook_key]
        projection_keys: list[tuple[Any, ...]] = []
        projection_dense: list[torch.Tensor] = []
        for expert in range(experts):
            _projection, tier, codes, scales, _key, _packed = rows[expert]
            identity = first._kmajor_cache_identity(
                projection, tier, expert, codebook_key, codebook
            )
            key = kmajor_module._base_key(codebook, codes, scales, identity)
            dense = kmajor_module._CACHE.get(key)
            if dense is None:
                raise RuntimeError(
                    f"layer graph missing prefilled tile projection={projection} expert={expert}"
                )
            projection_keys.append(key)
            projection_dense.append(dense)
        codes_slab = torch.stack([rows[expert][2] for expert in range(experts)])
        scales_slab = torch.stack([rows[expert][3] for expert in range(experts)])
        dense_slab = torch.stack(projection_dense)
        for expert, key in enumerate(projection_keys):
            kmajor_module._CACHE[key] = dense_slab[expert]
        kmajor_module._SENTINEL["cache_hits"] += len(projection_keys)
        shared[projection] = {
            "codebook_key": codebook_key,
            "codes": codes_slab,
            "scales": scales_slab,
            "dense": dense_slab,
            "tiers": sorted(tiers),
        }
        cache_keys.extend(projection_keys)
        dense_values.extend(projection_dense)
        del rows, projection_dense
    for layer in layers:
        layer._banana_smasher_graph_payloads = {
            projection: {
                "codebook": layer.codebooks[row["codebook_key"]],
                "codes": row["codes"],
                "scales": row["scales"],
                "dense": row["dense"],
            }
            for projection, row in shared.items()
        }
    del cache_keys, dense_values
    torch.cuda.empty_cache()
    return {
        "layers": len(layers),
        "projections": len(shared),
        "experts_per_projection": experts,
        "graph_nodes_per_layer": len(shared),
        "dense_slab_bytes": sum(
            int(row["dense"].numel() * row["dense"].element_size())
            for row in shared.values()
        ),
        "codes_slab_bytes": sum(
            int(row["codes"].numel() * row["codes"].element_size())
            for row in shared.values()
        ),
        "scales_slab_bytes": sum(
            int(row["scales"].numel() * row["scales"].element_size())
            for row in shared.values()
        ),
    }
