from __future__ import annotations

import pytest

import banana_smasher.update as update_module
from banana_smasher.update import (
    _activate_local_preflight,
    _logical_segment_bounds,
    _logical_source_plan,
    _set_logical_training_extent,
)


class _Authority(dict):
    receipt_sha256 = "a" * 64
    manifest_sha256 = "b" * 64


class _PreflightRuntime:
    def __init__(self) -> None:
        self.call = None

    def assert_preflight_sealed_structural(self, receipt, **kwargs):
        self.call = (receipt, kwargs)
        return _Authority(
            cache_root="/cas",
            ordered_window_ids=[27, 38, 39, 43],
            network_forbidden_during_update=True,
        )


class _Overlay:
    def __init__(self) -> None:
        self.call = None

    def activate_local_only(self, receipt, authority) -> None:
        self.call = (receipt, authority)


def test_activate_local_preflight_authenticates_and_activates_before_compute() -> None:
    preflight, overlay = _PreflightRuntime(), _Overlay()
    identity = _activate_local_preflight(
        overlay,
        preflight,
        receipt_path="/tmp/preflight.json",
        expected_receipt_sha256="a" * 64,
        expected_manifest_sha256="b" * 64,
        migration_receipt_path="/tmp/migration.json",
        expected_migration_receipt_sha256="c" * 64,
        expected_task_id="t_source",
    )
    assert preflight.call is not None
    assert overlay.call is not None
    assert identity["ordered_window_ids"] == [27, 38, 39, 43]
    assert identity["network_forbidden_during_update"] is True


def test_activate_local_preflight_requires_every_pin() -> None:
    with pytest.raises(RuntimeError, match="expected_receipt_sha256"):
        _activate_local_preflight(
            _Overlay(),
            _PreflightRuntime(),
            receipt_path="/tmp/preflight.json",
            expected_receipt_sha256=None,
            expected_manifest_sha256="b" * 64,
            migration_receipt_path="/tmp/migration.json",
            expected_migration_receipt_sha256="c" * 64,
            expected_task_id="t_source",
        )


def test_logical_8192_window_is_exactly_eight_contiguous_1024_segments() -> None:
    bounds = _logical_segment_bounds(1024, 8)
    assert bounds == [(index * 1024, (index + 1) * 1024) for index in range(8)]
    assert bounds[0] == (0, 1024)
    assert bounds[-1] == (7168, 8192)


@pytest.mark.parametrize("tokens,segments", [(0, 8), (1025, 8), (1024, 0), (1024, 9)])
def test_logical_segment_geometry_fails_closed(tokens: int, segments: int) -> None:
    with pytest.raises(ValueError):
        _logical_segment_bounds(tokens, segments)


def test_logical_window_expands_legacy_1024_loader_extent() -> None:
    class TrainingModule:
        T_TRAIN = 1024

    module = TrainingModule()
    assert _set_logical_training_extent(module, 8192) == 1024
    assert module.T_TRAIN == 8192


def test_logical_8192_extent_spans_explicit_sparse_source_windows_exactly() -> None:
    corpus = [
        {"real_len": length, "token_ids": list(range(length))}
        for length in [2048, 7, 2048, 9, 2044, 11, 2048, 13, 2048]
    ]
    plan = _logical_source_plan(corpus, [0, 2, 4, 6, 8], 8192)
    assert plan == [(0, 2048), (2, 2048), (4, 2044), (6, 2048), (8, 4)]
    assert sum(take for _, take in plan) == 8192


def test_logical_source_plan_fails_closed_when_explicit_assets_are_short() -> None:
    corpus = [{"real_len": 2048, "token_ids": list(range(2048))} for _ in range(4)]
    with pytest.raises(RuntimeError, match="provide 4096 tokens"):
        _logical_source_plan(corpus, [0, 3], 8192)


def test_full_depth_eight_segment_run_selects_sealed_resident_prefill() -> None:
    assert update_module._resident_prefill_policy(43, 8) == "sealed-eight-segment-full-depth"
    assert update_module._resident_prefill_policy(1, 8) == "manual-one-layer"
    assert update_module._resident_prefill_policy(43, 1) == "layer-window-eviction"


def test_full_depth_residency_keeps_activation_checkpointing_enabled() -> None:
    full_depth = update_module._runtime_memory_policy(43, 8)
    one_layer = update_module._runtime_memory_policy(1, 8)

    assert full_depth == {
        "keep_planes_resident": "1",
        "pin_planes": "1",
        "evict": "0",
        "checkpoint": "1",
    }
    assert one_layer["keep_planes_resident"] == "1"
    assert one_layer["checkpoint"] == "0"


def test_full_depth_resident_controller_uses_public_seal_and_segment_brackets() -> None:
    class Surface:
        def __init__(self) -> None:
            self.calls = []

        def seal_resident_planes(self):
            self.calls.append(("seal",))
            return {"layers": 43, "entries": 1}

        def begin_kmajor_timed_segment(self, index):
            self.calls.append(("begin", index))
            return {"segment_index": index, "boundary": "begin"}

        def end_kmajor_timed_segment(self, index):
            self.calls.append(("end", index))
            return {"segment_index": index, "boundary": "end"}

    surface = Surface()
    policy = "sealed-eight-segment-full-depth"
    inventory = update_module._seal_prefilled_planes(surface, policy)
    begin = update_module._begin_timed_segment(surface, policy, 3)
    end = update_module._end_timed_segment(surface, policy, 3)

    assert inventory == {"layers": 43, "entries": 1}
    assert begin == {"segment_index": 3, "boundary": "begin"}
    assert end == {"segment_index": 3, "boundary": "end"}
    assert surface.calls == [("seal",), ("begin", 3), ("end", 3)]


def test_segment_progress_receipt_is_durable_and_cumulative(tmp_path) -> None:
    receipt = tmp_path / "update.json"
    first = {"segment_index": 0, "forward_seconds": 1.0, "backward_seconds": 2.0}
    second = {"segment_index": 1, "forward_seconds": 1.5, "backward_seconds": 2.5}

    progress_path = update_module._seal_segment_progress(
        receipt, [first], logical_items=8192, segments=8
    )
    update_module._seal_segment_progress(
        receipt, [first, second], logical_items=8192, segments=8
    )

    payload = __import__("json").loads(progress_path.read_text())
    assert progress_path.name == "update.progress.json"
    assert payload["status"] == "RUNNING"
    assert payload["completed_segments"] == 2
    assert payload["expected_segments"] == 8
    assert payload["logical_items"] == 8192
    assert payload["segment_phases"] == [first, second]


def test_full_depth_corrected_guard_flags_60_gib_but_accepts_to_100_gib() -> None:
    acceptance = update_module._runtime_memory_acceptance(
        43,
        48 * 1024**3,
        60 * 1024**3 + 148 * 1024**2,
    )

    assert acceptance["hard_pass"] is True
    assert acceptance["minimum_mem_available_hard_floor_bytes"] == 16 * 1024**3
    assert acceptance["maximum_device_used_hard_ceiling_bytes"] == 100 * 1024**3
    assert acceptance["device_used_target_is_flag_only"] is True
    assert acceptance["device_used_target_flag"] == "FLAG_AT_OR_ABOVE_60_GIB_TARGET"
    assert acceptance["device_used_target_delta_bytes"] == 148 * 1024**2


def test_full_depth_corrected_guard_fails_only_at_hard_limits() -> None:
    assert update_module._runtime_memory_acceptance(43, 16 * 1024**3 - 1, 60 * 1024**3)[
        "hard_pass"
    ] is False
    assert update_module._runtime_memory_acceptance(43, 16 * 1024**3, 80 * 1024**3)[
        "hard_pass"
    ] is True
    assert update_module._runtime_memory_acceptance(43, 16 * 1024**3, 100 * 1024**3)[
        "hard_pass"
    ] is True
    assert update_module._runtime_memory_acceptance(43, 16 * 1024**3, 100 * 1024**3 + 1)[
        "hard_pass"
    ] is False