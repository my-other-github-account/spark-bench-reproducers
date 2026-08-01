from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from banana_smasher.cli import _parser, main


def test_solve_defaults_to_exact_accelerated_backend_without_runtime_proof() -> None:
    args = _parser().parse_args(
        [
            "solve",
            "--source-root",
            "/tmp/source",
            "--output",
            "/tmp/output",
        ]
    )

    assert args.backend == "exact-gemm"
    assert args.reference_search is False
    assert args.verbose_receipts is False
    assert not hasattr(args, "self_check_rows")
    assert not hasattr(args, "reference_fallback")


def test_solve_hides_developer_flags_from_user_help() -> None:
    parser = _parser()
    command_action = next(
        action for action in parser._actions if getattr(action, "choices", None)
    )
    solve = command_action.choices["solve"]
    help_text = solve.format_help()

    assert "--reference-search" not in help_text
    assert "--verbose-receipts" not in help_text
    assert "--implementation" not in help_text
    assert "--self-check" not in help_text


def _write_solve_fixture(root: Path) -> None:
    root.mkdir()
    np.save(
        root / "vectors.npy",
        np.asarray([[0.0, 0.0, 0.0, 0.0], [0.9, 1.0, 1.1, 1.0]], dtype=np.float32),
    )
    np.save(
        root / "codebook.npy",
        np.asarray(
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0],
                [-1.0, -1.0, -1.0, -1.0],
            ],
            dtype=np.float32,
        ),
    )
    (root / "solve.json").write_text(
        json.dumps(
            {
                "schema": "banana-smasher-solve-input-v1",
                "layer": 23,
                "cells": [
                    {
                        "cell": "L023.E000.P13",
                        "vectors": "vectors.npy",
                        "codebook": "codebook.npy",
                    }
                ],
            }
        )
    )


def test_reference_search_is_explicit_and_writes_slim_receipt(
    tmp_path: Path, capsys
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_solve_fixture(source)

    assert (
        main(
            [
                "solve",
                "--source-root",
                str(source),
                "--output",
                str(output),
                "--device",
                "cpu",
                "--reference-search",
            ]
        )
        == 0
    )

    summary = json.loads(capsys.readouterr().out)
    assert set(summary) == {
        "artifact",
        "backend",
        "command",
        "elapsed_seconds",
        "receipt",
        "status",
    }
    assert summary["status"] == "PASS"
    assert summary["backend"] == "reference-search"
    assert np.load(output / "winners.npz", allow_pickle=False)["L023.E000.P13"].tolist() == [0, 1]
    receipt = json.loads((output / "SOLVE_RECEIPT.json").read_text())
    assert receipt["shape"] == {"candidates": 3, "cells": 1, "rows": 2}
    assert "self_check" not in json.dumps(receipt)
    assert "fallback" not in json.dumps(receipt)


def _add_frozen_bucket(root: Path) -> None:
    np.save(root / "weights.npy", np.asarray([[0.0] * 32, [1.0] * 32], dtype=np.float32))
    np.save(root / "h.npy", np.ones((32,), dtype=np.float32))
    np.save(root / "codes.npy", np.zeros((2, 2, 8), dtype=np.int16))
    np.save(root / "scales.npy", np.full((2, 2, 1), 127, dtype=np.uint8))
    np.save(
        root / "bucket_codebooks.npy",
        np.asarray([[0.0] * 4, [1.0] * 4], dtype=np.float32),
    )
    np.save(root / "bucket_offsets.npy", np.asarray([0, 1], dtype=np.int64))
    manifest = json.loads((root / "solve.json").read_text())
    manifest["cells"][0]["frozen_bucket"] = {
        "weights": "weights.npy",
        "h": "h.npy",
        "codes": "codes.npy",
        "scales": "scales.npy",
        "codebooks": "bucket_codebooks.npy",
        "codebook_offsets": "bucket_offsets.npy",
        "vector_width": 4,
        "options": ["zero", "one"],
    }
    (root / "solve.json").write_text(json.dumps(manifest))


def test_solve_scores_all_frozen_bucket_options_and_selects_ties_deterministically(
    tmp_path: Path, capsys
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_solve_fixture(source)
    _add_frozen_bucket(source)

    assert main(
        [
            "solve",
            "--source-root",
            str(source),
            "--output",
            str(output),
            "--device",
            "cpu",
            "--reference-search",
        ]
    ) == 0
    capsys.readouterr()

    scores = np.load(output / "bucket_scores.npz", allow_pickle=False)
    assert scores["L023.E000.P13"].tolist() == [32.0, 32.0]
    receipt = json.loads((output / "SOLVE_RECEIPT.json").read_text())
    assert receipt["buckets"] == [
        {
            "cell": "L023.E000.P13",
            "options": ["zero", "one"],
            "winner": "zero",
            "winner_index": 0,
        }
    ]
    assert receipt["bucket_artifact"] == str(output / "bucket_scores.npz")


def test_default_solve_fails_loud_without_accelerator_and_publishes_nothing(
    tmp_path: Path, capsys
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_solve_fixture(source)

    assert (
        main(
            [
                "solve",
                "--source-root",
                str(source),
                "--output",
                str(output),
                "--device",
                "cpu",
            ]
        )
        == 2
    )

    failure = json.loads(capsys.readouterr().err)
    assert failure["status"] == "FAIL"
    assert "no reference fallback" in failure["error"]
    assert not output.exists()
