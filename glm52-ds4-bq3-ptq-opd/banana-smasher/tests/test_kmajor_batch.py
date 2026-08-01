from __future__ import annotations

import torch

from banana_smasher.kmajor_fused import (
    _grouped_vjp_launch_grid,
    fused_codebook_vjp,
    fused_codebook_vjp_from_inputs,
    fused_grouped_codebook_vjp,
    fused_grouped_codebook_vjp_from_inputs,
)
from banana_smasher.kmajor_batch import (
    BatchedKMajorVQLinearFn,
    batched_kmajor_vjp_stats,
    reset_batched_kmajor_vjp,
)
from banana_smasher.kmajor_graph import (
    LayerProjectionKMajorFn,
    layer_graph_forward,
    layer_graph_vjp_stats,
    reset_layer_graph_vjp,
)


class LegacyKMajorVQLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, codebook32, codes, scales, dense_kn):
        ctx.save_for_backward(x, codebook32, codes, scales, dense_kn)
        ctx.codebook_shape = tuple(codebook32.shape)
        return torch.bmm(x.unsqueeze(0), dense_kn.unsqueeze(0)).squeeze(0)

    @staticmethod
    def backward(ctx, grad_out):
        x, _codebook32, codes, scales, dense_kn = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        grad_x = torch.mm(grad_out, dense_kn.transpose(0, 1))
        grad_weight_nk = torch.mm(grad_out.transpose(0, 1), x)
        d = int(ctx.codebook_shape[1])
        scale_columns = torch.exp2(scales.float() - 127.0).repeat_interleave(32, -1)
        grouped = (grad_weight_nk.float() * scale_columns).reshape(
            codes.shape[0], codes.shape[1], d
        )
        grad_codebook = torch.zeros(
            ctx.codebook_shape, dtype=torch.float32, device=grad_out.device
        )
        grad_codebook.index_add_(
            0, codes.reshape(-1).long(), grouped.reshape(-1, d)
        )
        return grad_x, grad_codebook, None, None, None


def _dense(codebook, codes, scales):
    wire = codebook.detach().to(torch.float16).float()
    scale_columns = torch.exp2(scales.float() - 127.0).repeat_interleave(32, -1)
    weight_nk = (
        wire[codes.long()].reshape(codes.shape[0], codes.shape[1] * codebook.shape[1])
        * scale_columns
    )
    return weight_nk.transpose(0, 1).contiguous()


def test_batched_kmajor_vjp_matches_legacy_shared_codebook_gradient():
    torch.manual_seed(1436)
    experts, rows, in_features, d, k, m = 4, 4, 32, 2, 8, 3
    codebook_ref = torch.randn(k, d, dtype=torch.float32, requires_grad=True)
    codebook_new = codebook_ref.detach().clone().requires_grad_(True)
    codes = [torch.randint(0, k, (rows, in_features // d), dtype=torch.int32) for _ in range(experts)]
    scales = [torch.full((rows, in_features // 32), 127, dtype=torch.uint8) for _ in range(experts)]
    xs_ref = [torch.randn(m, in_features, dtype=torch.float32, requires_grad=True) for _ in range(experts)]
    xs_new = [x.detach().clone().requires_grad_(True) for x in xs_ref]
    output_grads = [torch.randn(m, rows, dtype=torch.float32) for _ in range(experts)]

    ref_outputs = [
        LegacyKMajorVQLinearFn.apply(x, codebook_ref, code, scale, _dense(codebook_ref, code, scale))
        for x, code, scale in zip(xs_ref, codes, scales)
    ]
    sum((out * grad).sum() for out, grad in zip(ref_outputs, output_grads)).backward()

    reset_batched_kmajor_vjp(batch_size=2)
    new_outputs = [
        BatchedKMajorVQLinearFn.apply(x, codebook_new, code, scale, _dense(codebook_new, code, scale))
        for x, code, scale in zip(xs_new, codes, scales)
    ]
    sum((out * grad).sum() for out, grad in zip(new_outputs, output_grads)).backward()

    for observed, expected in zip(new_outputs, ref_outputs):
        torch.testing.assert_close(observed, expected)
    for observed, expected in zip(xs_new, xs_ref):
        torch.testing.assert_close(observed.grad, expected.grad)
    torch.testing.assert_close(codebook_new.grad, codebook_ref.grad)
    stats = batched_kmajor_vjp_stats()
    assert stats["forward_calls"] == experts
    assert stats["backward_calls"] == experts
    assert stats["batch_flushes"] == 2
    assert stats["unique_groups"] == 1
    assert stats["max_pending"] == 2


def test_batched_kmajor_vjp_executes_cuda_batch_flushes():
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    torch.manual_seed(1436)
    codebook = torch.randn(8, 2, device=device, dtype=torch.float32, requires_grad=True)
    xs = [
        torch.randn(3, 32, device=device, dtype=torch.float32, requires_grad=True)
        for _ in range(4)
    ]
    codes = [
        torch.randint(0, 8, (4, 16), device=device, dtype=torch.int32)
        for _ in range(4)
    ]
    scales = [
        torch.full((4, 1), 127, device=device, dtype=torch.uint8)
        for _ in range(4)
    ]
    reset_batched_kmajor_vjp(batch_size=2)
    outputs = [
        BatchedKMajorVQLinearFn.apply(
            x, codebook, code, scale, _dense(codebook, code, scale)
        )
        for x, code, scale in zip(xs, codes, scales)
    ]
    sum(output.float().square().mean() for output in outputs).backward()
    torch.cuda.synchronize()
    assert codebook.grad is not None
    assert bool(torch.isfinite(codebook.grad).all())
    assert all(x.grad is not None and bool(torch.isfinite(x.grad).all()) for x in xs)
    stats = batched_kmajor_vjp_stats()
    assert stats["batch_flushes"] == 2
    assert stats["active_groups"] == 0


def test_batched_kmajor_vjp_ignores_checkpoint_forward_without_backward():
    torch.manual_seed(1436)
    codebook = torch.randn(8, 2, dtype=torch.float32, requires_grad=True)
    codes = [torch.randint(0, 8, (4, 16), dtype=torch.int32) for _ in range(4)]
    scales = [torch.full((4, 1), 127, dtype=torch.uint8) for _ in range(4)]
    xs = [torch.randn(3, 32, dtype=torch.float32, requires_grad=True) for _ in range(4)]
    grads = [torch.randn(3, 4, dtype=torch.float32) for _ in range(4)]
    reset_batched_kmajor_vjp(batch_size=2)

    # Non-reentrant activation checkpointing executes one forward whose custom
    # nodes are not traversed, then recomputes the same calls for backward.
    unused = [
        BatchedKMajorVQLinearFn.apply(
            x, codebook, code, scale, _dense(codebook, code, scale)
        )
        for x, code, scale in zip(xs, codes, scales)
    ]
    assert len(unused) == 4
    used = [
        BatchedKMajorVQLinearFn.apply(
            x, codebook, code, scale, _dense(codebook, code, scale)
        )
        for x, code, scale in zip(xs, codes, scales)
    ]
    sum((output * grad).sum() for output, grad in zip(used, grads)).backward()

    assert codebook.grad is not None
    assert bool(torch.isfinite(codebook.grad).all())
    stats = batched_kmajor_vjp_stats()
    assert stats["forward_calls"] == 8
    assert stats["backward_calls"] == 4
    assert stats["batch_flushes"] == 2
    assert stats["active_groups"] == 0


def test_layer_projection_graph_vjp_matches_legacy_expert_nodes():
    torch.manual_seed(1436)
    experts, rows, in_features, d, k, m = 4, 4, 32, 2, 8, 3
    codebook_ref = torch.randn(k, d, dtype=torch.float32, requires_grad=True)
    codebook_new = codebook_ref.detach().clone().requires_grad_(True)
    codes = torch.stack([
        torch.randint(0, k, (rows, in_features // d), dtype=torch.int32)
        for _ in range(experts)
    ])
    scales = torch.full(
        (experts, rows, in_features // 32), 127, dtype=torch.uint8
    )
    xs_ref = [
        torch.randn(m, in_features, dtype=torch.float32, requires_grad=True)
        for _ in range(experts)
    ]
    x_new = torch.stack([value.detach() for value in xs_ref]).requires_grad_(True)
    dense = torch.stack([
        _dense(codebook_ref, codes[index], scales[index])
        for index in range(experts)
    ])
    output_grad = torch.randn(experts, m, rows, dtype=torch.float32)

    ref_outputs = [
        LegacyKMajorVQLinearFn.apply(
            x,
            codebook_ref,
            codes[index],
            scales[index],
            dense[index],
        )
        for index, x in enumerate(xs_ref)
    ]
    sum(
        (output * output_grad[index]).sum()
        for index, output in enumerate(ref_outputs)
    ).backward()

    reset_layer_graph_vjp()
    observed = LayerProjectionKMajorFn.apply(
        x_new, codebook_new, codes, scales, dense
    )
    (observed * output_grad).sum().backward()

    torch.testing.assert_close(observed, torch.stack(ref_outputs))
    torch.testing.assert_close(x_new.grad, torch.stack([value.grad for value in xs_ref]))
    torch.testing.assert_close(codebook_new.grad, codebook_ref.grad)
    stats = layer_graph_vjp_stats()
    assert stats["forward_calls"] == 1
    assert stats["backward_calls"] == 1
    assert stats["grouped_experts"] == experts
    assert stats["max_nodes_per_projection"] == 1
    assert stats["grad_weight_bmm_launches"] == 1
    assert stats["reduction_kernel_launches"] == 0


def test_layer_graph_forward_matches_balanced_expert_loop():
    torch.manual_seed(1436)
    experts, tokens, top_k, hidden, d, k = 4, 8, 2, 32, 2, 8
    routes_per_expert = tokens * top_k // experts
    route_index = torch.tensor(
        [[(token + slot * 2) % experts for slot in range(top_k)] for token in range(tokens)]
    )
    assert torch.equal(
        torch.bincount(route_index.reshape(-1), minlength=experts),
        torch.full((experts,), routes_per_expert),
    )
    route_weights = torch.rand(tokens, top_k)
    hidden_ref = torch.randn(tokens, hidden, requires_grad=True)
    hidden_new = hidden_ref.detach().clone().requires_grad_(True)
    codebook13_ref = torch.randn(k, d, requires_grad=True)
    codebook2_ref = torch.randn(k, d, requires_grad=True)
    codebook13_new = codebook13_ref.detach().clone().requires_grad_(True)
    codebook2_new = codebook2_ref.detach().clone().requires_grad_(True)
    rows13, rows2 = hidden * 2, hidden
    codes13 = torch.randint(0, k, (experts, rows13, hidden // d), dtype=torch.int32)
    codes2 = torch.randint(0, k, (experts, rows2, hidden // d), dtype=torch.int32)
    scales13 = torch.full((experts, rows13, hidden // 32), 127, dtype=torch.uint8)
    scales2 = torch.full((experts, rows2, hidden // 32), 127, dtype=torch.uint8)
    dense13 = torch.stack([
        _dense(codebook13_ref, codes13[index], scales13[index])
        for index in range(experts)
    ])
    dense2 = torch.stack([
        _dense(codebook2_ref, codes2[index], scales2[index])
        for index in range(experts)
    ])

    final_ref = torch.zeros_like(hidden_ref)
    for expert in range(experts):
        slot, token = torch.where(route_index.transpose(0, 1) == expert)
        current = LegacyKMajorVQLinearFn.apply(
            hidden_ref[token], codebook13_ref, codes13[expert], scales13[expert], dense13[expert]
        )
        gate, up = current.chunk(2, dim=-1)
        intermediate = torch.nn.functional.silu(gate.clamp(max=10.0)) * up.clamp(
            min=-10.0, max=10.0
        )
        current = LegacyKMajorVQLinearFn.apply(
            intermediate, codebook2_ref, codes2[expert], scales2[expert], dense2[expert]
        )
        final_ref.index_add_(
            0, token, current * route_weights[token, slot, None]
        )

    payloads = {
        "13": {
            "codebook": codebook13_new,
            "codes": codes13,
            "scales": scales13,
            "dense": dense13,
        },
        "2": {
            "codebook": codebook2_new,
            "codes": codes2,
            "scales": scales2,
            "dense": dense2,
        },
    }
    final_new = layer_graph_forward(
        hidden_new, route_index, route_weights, payloads, limit=10.0
    )
    grad = torch.randn_like(final_ref)
    (final_ref * grad).sum().backward()
    (final_new * grad).sum().backward()

    torch.testing.assert_close(final_new, final_ref)
    torch.testing.assert_close(hidden_new.grad, hidden_ref.grad)
    torch.testing.assert_close(codebook13_new.grad, codebook13_ref.grad)
    torch.testing.assert_close(codebook2_new.grad, codebook2_ref.grad)


def _run_layer_graph_one_layer_adam_parity() -> dict[str, float]:
    torch.manual_seed(1436)
    experts, tokens, top_k, hidden, d, k = 4, 8, 2, 32, 2, 8
    route_index = torch.tensor(
        [[(token + slot * 2) % experts for slot in range(top_k)] for token in range(tokens)]
    )
    route_weights = torch.rand(tokens, top_k)
    hidden_ref = torch.randn(tokens, hidden, requires_grad=True)
    hidden_new = hidden_ref.detach().clone().requires_grad_(True)
    codebook13_ref = torch.randn(k, d, requires_grad=True)
    codebook2_ref = torch.randn(k, d, requires_grad=True)
    codebook13_new = codebook13_ref.detach().clone().requires_grad_(True)
    codebook2_new = codebook2_ref.detach().clone().requires_grad_(True)
    rows13, rows2 = hidden * 2, hidden
    codes13 = torch.randint(0, k, (experts, rows13, hidden // d), dtype=torch.int32)
    codes2 = torch.randint(0, k, (experts, rows2, hidden // d), dtype=torch.int32)
    scales13 = torch.full((experts, rows13, hidden // 32), 127, dtype=torch.uint8)
    scales2 = torch.full((experts, rows2, hidden // 32), 127, dtype=torch.uint8)
    dense13 = torch.stack(
        [_dense(codebook13_ref, codes13[index], scales13[index]) for index in range(experts)]
    )
    dense2 = torch.stack(
        [_dense(codebook2_ref, codes2[index], scales2[index]) for index in range(experts)]
    )

    final_ref = torch.zeros_like(hidden_ref)
    for expert in range(experts):
        slot, token = torch.where(route_index.transpose(0, 1) == expert)
        current = LegacyKMajorVQLinearFn.apply(
            hidden_ref[token],
            codebook13_ref,
            codes13[expert],
            scales13[expert],
            dense13[expert],
        )
        gate, up = current.chunk(2, dim=-1)
        intermediate = torch.nn.functional.silu(gate.clamp(max=10.0)) * up.clamp(
            min=-10.0, max=10.0
        )
        current = LegacyKMajorVQLinearFn.apply(
            intermediate,
            codebook2_ref,
            codes2[expert],
            scales2[expert],
            dense2[expert],
        )
        final_ref.index_add_(
            0, token, current * route_weights[token, slot, None]
        )

    final_new = layer_graph_forward(
        hidden_new,
        route_index,
        route_weights,
        {
            "13": {
                "codebook": codebook13_new,
                "codes": codes13,
                "scales": scales13,
                "dense": dense13,
            },
            "2": {
                "codebook": codebook2_new,
                "codes": codes2,
                "scales": scales2,
                "dense": dense2,
            },
        },
        limit=10.0,
    )
    loss_ref = final_ref.float().square().mean()
    loss_new = final_new.float().square().mean()
    loss_ref.backward()
    loss_new.backward()
    assert hidden_ref.grad is not None and hidden_new.grad is not None
    assert codebook13_ref.grad is not None and codebook13_new.grad is not None
    assert codebook2_ref.grad is not None and codebook2_new.grad is not None

    metrics = {
        "loss_abs_delta": float((loss_new.detach() - loss_ref.detach()).abs()),
        "input_grad_max_abs": float((hidden_new.grad - hidden_ref.grad).abs().max()),
        "projection_13_grad_max_abs": float(
            (codebook13_new.grad - codebook13_ref.grad).abs().max()
        ),
        "projection_2_grad_max_abs": float(
            (codebook2_new.grad - codebook2_ref.grad).abs().max()
        ),
    }

    torch.testing.assert_close(loss_new, loss_ref)
    torch.testing.assert_close(hidden_new.grad, hidden_ref.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(codebook13_new.grad, codebook13_ref.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(codebook2_new.grad, codebook2_ref.grad, rtol=1e-5, atol=1e-6)

    optimizer_ref = torch.optim.Adam([codebook13_ref, codebook2_ref], lr=1e-3)
    optimizer_new = torch.optim.Adam([codebook13_new, codebook2_new], lr=1e-3)
    optimizer_ref.step()
    optimizer_new.step()
    metrics["projection_13_parameter_max_abs"] = float(
        (codebook13_new.detach() - codebook13_ref.detach()).abs().max()
    )
    metrics["projection_2_parameter_max_abs"] = float(
        (codebook2_new.detach() - codebook2_ref.detach()).abs().max()
    )
    torch.testing.assert_close(codebook13_new, codebook13_ref, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(codebook2_new, codebook2_ref, rtol=1e-5, atol=1e-6)
    return metrics


def test_layer_graph_one_layer_adam_step_matches_legacy():
    metrics = _run_layer_graph_one_layer_adam_parity()
    assert metrics["loss_abs_delta"] <= 1e-6
    assert metrics["input_grad_max_abs"] <= 1e-6
    assert metrics["projection_13_grad_max_abs"] <= 5e-4
    assert metrics["projection_2_grad_max_abs"] <= 5e-4
    assert metrics["projection_13_parameter_max_abs"] <= 1e-6
    assert metrics["projection_2_parameter_max_abs"] <= 1e-6


def test_fused_cuda_codebook_vjp_matches_eager_reduction():
    if not torch.cuda.is_available():
        return
    torch.manual_seed(1436)
    device = torch.device("cuda")
    rows, in_features, d, k = 64, 32, 2, 8
    grad_weight = torch.randn(
        rows, in_features, device=device, dtype=torch.bfloat16
    )
    codes = torch.randint(
        0, k, (rows, in_features // d), device=device, dtype=torch.int32
    )
    scales = torch.randint(
        124, 130, (rows, in_features // 32), device=device, dtype=torch.uint8
    )
    scale_columns = torch.exp2(scales.float() - 127.0).repeat_interleave(32, -1)
    grouped = (grad_weight.float() * scale_columns).reshape(
        rows, in_features // d, d
    )
    expected = torch.zeros(k, d, device=device, dtype=torch.float32)
    expected.index_add_(0, codes.reshape(-1).long(), grouped.reshape(-1, d))

    observed = fused_codebook_vjp(grad_weight, codes, scales, k, d)
    torch.cuda.synchronize()
    torch.testing.assert_close(observed, expected, rtol=2e-5, atol=2e-4)


def test_fused_cuda_codebook_vjp_from_inputs_matches_eager_reduction():
    if not torch.cuda.is_available():
        return
    torch.manual_seed(1436)
    device = torch.device("cuda")
    m, rows, in_features, d, k = 3, 64, 32, 2, 8
    grad_out = torch.randn(m, rows, device=device, dtype=torch.bfloat16)
    activation = torch.randn(m, in_features, device=device, dtype=torch.bfloat16)
    codes = torch.randint(
        0, k, (rows, in_features // d), device=device, dtype=torch.int32
    )
    scales = torch.randint(
        124, 130, (rows, in_features // 32), device=device, dtype=torch.uint8
    )
    grad_weight = torch.mm(grad_out.transpose(0, 1), activation)
    scale_columns = torch.exp2(scales.float() - 127.0).repeat_interleave(32, -1)
    grouped = (grad_weight.float() * scale_columns).reshape(
        rows, in_features // d, d
    )
    expected = torch.zeros(k, d, device=device, dtype=torch.float32)
    expected.index_add_(0, codes.reshape(-1).long(), grouped.reshape(-1, d))

    observed = fused_codebook_vjp_from_inputs(
        grad_out, activation, codes, scales, k, d
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(observed, expected, rtol=5e-3, atol=5e-3)


def test_grouped_vjp_launch_grid_uses_expert_axis_for_production_shape():
    grid = _grouped_vjp_launch_grid(
        experts=256,
        rows=4096,
        in_features=4096,
        block=256,
    )

    assert grid == (65536, 256)
    assert (grid[0] - 1) * 256 + 255 < 2**31


def test_fused_cuda_grouped_codebook_vjp_matches_expert_sum():
    if not torch.cuda.is_available():
        return
    torch.manual_seed(1436)
    device = torch.device("cuda")
    experts, m, rows, in_features, d, k = 4, 3, 64, 32, 2, 8
    grad_out = torch.randn(
        experts, m, rows, device=device, dtype=torch.bfloat16
    )
    activation = torch.randn(
        experts, m, in_features, device=device, dtype=torch.bfloat16
    )
    codes = torch.randint(
        0,
        k,
        (experts, rows, in_features // d),
        device=device,
        dtype=torch.int32,
    )
    scales = torch.randint(
        124,
        130,
        (experts, rows, in_features // 32),
        device=device,
        dtype=torch.uint8,
    )
    expected = torch.zeros(k, d, device=device, dtype=torch.float32)
    for expert in range(experts):
        expected.add_(
            fused_codebook_vjp_from_inputs(
                grad_out[expert],
                activation[expert],
                codes[expert],
                scales[expert],
                k,
                d,
            )
        )
    observed = fused_grouped_codebook_vjp_from_inputs(
        grad_out, activation, codes, scales, k, d
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(observed, expected, rtol=5e-3, atol=5e-3)


def test_fused_cuda_grouped_codebook_vjp_from_weights_matches_expert_sum():
    if not torch.cuda.is_available():
        return
    torch.manual_seed(1437)
    device = torch.device("cuda")
    experts, rows, in_features, d, k = 33, 64, 32, 2, 8
    grad_weight = torch.randn(
        experts, rows, in_features, device=device, dtype=torch.bfloat16
    )
    codes = torch.randint(
        0, k, (experts, rows, in_features // d), device=device, dtype=torch.int32
    )
    scales = torch.randint(
        124, 130, (experts, rows, in_features // 32), device=device, dtype=torch.uint8
    )
    expected = torch.zeros(k, d, device=device, dtype=torch.float32)
    for expert in range(experts):
        expected.add_(
            fused_codebook_vjp(
                grad_weight[expert], codes[expert], scales[expert], k, d
            )
        )
    observed = fused_grouped_codebook_vjp(grad_weight, codes, scales, k, d)
    torch.cuda.synchronize()
    torch.testing.assert_close(observed, expected, rtol=5e-3, atol=5e-3)
