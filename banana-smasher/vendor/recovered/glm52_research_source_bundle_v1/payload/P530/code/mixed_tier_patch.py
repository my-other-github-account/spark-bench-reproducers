"""Environment-gated heterogeneous expert adapter for vLLM DeepseekV4."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import re
import sys
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

_INSTALLED = False
_ORIGINAL_INIT = None
_ORIGINAL_MODEL_LOAD = None


def _load_backend():
    backend_path = Path(os.environ["MIXED_TIER_BACKEND"])
    spec = importlib.util.spec_from_file_location("mixed_tier_backend_runtime", backend_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _layer_index(prefix: str) -> int:
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", prefix)
    if not match:
        raise ValueError(f"cannot extract DeepseekV4 layer index from {prefix!r}")
    return int(match.group(1))


class MixedTierExpertsAdapter(nn.Module):
    is_internal_router = False

    def __init__(self, artifact_path: str, backend: Any, layer_index: int):
        super().__init__()
        self.layer_index = int(layer_index)
        self.artifact_path = artifact_path
        self.mixed = backend.MixedTierLayer.from_file(artifact_path, layer_index, device="cuda")
        self.route_calls = 0
        self.routed_tokens = 0
        self.expert_pairs = 0

    @property
    def resident_bytes(self) -> int:
        return self.mixed.resident_bytes

    def forward_packed(self, hidden_states: torch.Tensor, topk_weights: torch.Tensor,
                       topk_ids: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape
        flat = hidden_states.reshape(-1, original_shape[-1])
        tokens, hidden = flat.shape
        topk = topk_ids.shape[-1]
        if hidden != 4096 or topk != 6:
            raise ValueError(f"expected DS4 hidden=4096/topk=6, got {hidden}/{topk}")
        expanded = flat[:, None, :].expand(tokens, topk, hidden).reshape(tokens * topk, hidden)
        ids = topk_ids.reshape(-1).to(device=flat.device, dtype=torch.long)
        fused = self.mixed.forward(expanded, ids, "fused13")
        gate, up = fused.chunk(2, dim=-1)
        # Serving values are intentionally uncalibrated. Keep the performance
        # path finite without changing which tier-specific kernels execute.
        activated = F.silu(gate.clamp(min=-10, max=10)) * up.clamp(min=-10, max=10)
        down = self.mixed.forward(activated, ids, "down")
        mixed = (down.reshape(tokens, topk, hidden) *
                 topk_weights[..., None].to(down.dtype)).sum(dim=1)
        self.route_calls += 1
        self.routed_tokens += tokens
        self.expert_pairs += tokens * topk
        return mixed.reshape(original_shape)

    def counters(self) -> dict[str, Any]:
        return {
            "route_calls": self.route_calls,
            "routed_tokens": self.routed_tokens,
            "expert_pairs": self.expert_pairs,
            "dispatch": self.mixed.counters(),
        }


def _score_and_topk(moe, router_logits: torch.Tensor):
    if moe.scoring_func == "sqrtsoftplus":
        scores = torch.sqrt(F.softplus(router_logits.float()))
    elif moe.scoring_func == "sigmoid":
        scores = torch.sigmoid(router_logits.float())
    elif moe.scoring_func == "softmax":
        scores = torch.softmax(router_logits.float(), dim=-1)
    else:
        raise NotImplementedError(moe.scoring_func)
    correction = moe.gate.e_score_correction_bias
    selection = scores if correction is None else scores + correction.float()
    _, topk_ids = torch.topk(selection, k=moe.n_activated_experts, dim=-1, sorted=False)
    topk_weights = scores.gather(-1, topk_ids)
    if moe.renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    return topk_weights * moe.routed_scaling_factor, topk_ids


def install() -> None:
    global _INSTALLED, _ORIGINAL_INIT, _ORIGINAL_MODEL_LOAD
    if _INSTALLED:
        return
    from vllm.model_executor.models import deepseek_v4 as dv4
    _ORIGINAL_INIT = dv4.DeepseekV4MoE._init_fused_moe_experts
    _ORIGINAL_MODEL_LOAD = dv4.DeepseekV4Model.load_weights

    def mixed_init(self, config, quant_config, prefix):
        artifact = os.environ.get("MIXED_TIER_ARTIFACT")
        if not artifact:
            return _ORIGINAL_INIT(self, config, quant_config, prefix)
        layer = _layer_index(prefix)
        self.tp_rank = 0
        self.n_local_experts = config.n_routed_experts
        self.experts_start_idx = 0
        self.experts_end_idx = config.n_routed_experts
        self.experts = MixedTierExpertsAdapter(artifact, _load_backend(), layer)

    def mixed_forward(self, hidden_states, input_ids=None):
        if not isinstance(self.experts, MixedTierExpertsAdapter):
            return self._mixed_original_forward_fused_moe(hidden_states, input_ids)
        if self.gate.tid2eid is not None:
            if input_ids is None:
                raise ValueError("hash routing requires input_ids")
            topk_ids = self.gate.tid2eid[input_ids.reshape(-1).long()]
            topk_weights = torch.full(topk_ids.shape, 1.0 / topk_ids.shape[-1],
                                      device=hidden_states.device, dtype=torch.float32)
        else:
            router_logits, _ = self.gate(hidden_states)
            topk_weights, topk_ids = _score_and_topk(self, router_logits)
        return self.experts.forward_packed(hidden_states, topk_weights, topk_ids)

    def mixed_model_load(self, weights):
        if not os.environ.get("MIXED_TIER_ARTIFACT"):
            return _ORIGINAL_MODEL_LOAD(self, weights)
        def non_expert_weights():
            for name, tensor in weights:
                if ".experts." not in name:
                    yield name, tensor
        loaded = _ORIGINAL_MODEL_LOAD(self, non_expert_weights())
        loaded.add("mixed_tier_experts:env")
        return loaded

    dv4.DeepseekV4MoE._mixed_original_forward_fused_moe = dv4.DeepseekV4MoE._forward_fused_moe
    dv4.DeepseekV4MoE._init_fused_moe_experts = mixed_init
    dv4.DeepseekV4MoE._forward_fused_moe = mixed_forward
    dv4.DeepseekV4Model.load_weights = mixed_model_load
    _INSTALLED = True
