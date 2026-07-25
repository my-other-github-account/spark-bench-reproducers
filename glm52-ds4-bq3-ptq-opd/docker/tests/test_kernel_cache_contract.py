from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1] / "runtime" / "mixed_tier_backend.py"


def test_decode_row_count_is_runtime_not_a_triton_compile_key() -> None:
    source = BACKEND.read_text()

    assert "R: tl.constexpr" not in source
    for kernel in (
        "_qtip_gemv",
        "_truevq_d4_gemv",
        "_truevq_d8_gemv",
        "_native_mxfp4_gemv",
    ):
        assert f"def {kernel}" in source
