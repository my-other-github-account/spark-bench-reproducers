from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1] / "runtime" / "mixed_tier_backend.py"


def test_decode_row_count_uses_pointer_only_launcher_workaround() -> None:
    source = BACKEND.read_text()

    # On the shipped aarch64 Triton runtime, passing R as a runtime scalar
    # segfaults for multi-row MoE launches. R only sizes grid Y and is never
    # read in the kernels, so all four decode kernels deliberately compile it
    # as a constexpr to preserve the pointer-only launcher ABI.
    assert source.count("R: tl.constexpr") == 4
    assert "runtime-scalar launcher segfaults" in source
    for kernel in (
        "_qtip_gemv",
        "_truevq_d4_gemv",
        "_truevq_d8_gemv",
        "_native_mxfp4_gemv",
    ):
        assert f"def {kernel}" in source
