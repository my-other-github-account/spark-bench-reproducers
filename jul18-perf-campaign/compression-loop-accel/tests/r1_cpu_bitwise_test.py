#!/usr/bin/env python3
"""CPU bitwise pre-validation of the R1 bulk dequant chain (<source-task>).

Compares, on randomized VQ payloads:
  scalar path (sealed r5_fullmenu_source.expert()):
      per-row: cb.to(fp16).float()[codes.long()].reshape(N,-1)
               * exp2(sc.float()-127).repeat_interleave(32, dim=1) -> bf16
  bulk path (accel fill_layer):
      batched: cb.to(fp16).float()[codes.long()].reshape(B,N,-1)
               * exp2(sc.float()-127).repeat_interleave(32, dim=-1) -> bf16

Every op is elementwise/gather (no reductions), so results must be BITWISE
identical regardless of batch shape. This is the CPU analog of the 8w golden
gate; the on-host gate remains the authority for the end-to-end rail.
"""
import json
import sys
from pathlib import Path

import torch

torch.manual_seed(1234)

HERE = Path(__file__).resolve().parent
results = []


def check(name, ok, detail=""):
    results.append({"test": name, "pass": bool(ok), "detail": detail})
    print(("PASS " if ok else "FAIL ") + name + (f" | {detail}" if detail else ""))


for tier_name, k, d, n_rows, n_cols in [
    ("d8_k256", 256, 8, 4096, 4096 // 8),
    ("d8_k4096", 4096, 8, 4096, 4096 // 8),
    ("d4_k1024", 1024, 4, 4096, 4096 // 4),
]:
    codebook = (torch.randn(k, d, dtype=torch.float32) * 0.05)
    batch = 16
    codes = torch.randint(0, k, (batch, n_rows, n_cols // 1), dtype=torch.int32)
    # scales are uint8 exponent bytes; 32 columns share one scale
    scales = torch.randint(96, 160, (batch, n_rows, (n_cols * d) // 32),
                           dtype=torch.uint8)

    # scalar path (row-at-a-time, sealed op order)
    wire_scalar = codebook.to(torch.float16).float()
    outs = []
    for row in range(batch):
        scale_columns = torch.exp2(
            scales[row].float() - 127.0
        ).repeat_interleave(32, dim=1)
        weights = wire_scalar[codes[row].long()].reshape(
            codes[row].shape[0], -1
        ) * scale_columns
        outs.append(weights.to(torch.bfloat16))
    scalar = torch.stack(outs)

    # bulk path (batched, accel op order)
    wire_bulk = codebook.to(torch.float16).float()
    scale_columns = torch.exp2(scales.float() - 127.0).repeat_interleave(32, dim=-1)
    bulk = (
        wire_bulk[codes.long()].reshape(codes.shape[0], codes.shape[1], -1)
        * scale_columns
    ).to(torch.bfloat16)

    bitwise = torch.equal(
        scalar.view(torch.uint16) if hasattr(scalar, "view") else scalar, 
        bulk.view(torch.uint16),
    )
    check(f"bitwise_{tier_name}", bitwise,
          f"batch={batch} rows={n_rows} shape={tuple(bulk.shape)}")

    # sub-batching invariance (FULLMENU_ASSEMBLY_BATCH boundary effects)
    for sub in (1, 3, 8):
        pieces = []
        for start in range(0, batch, sub):
            piece_codes = codes[start:start + sub]
            piece_scales = scales[start:start + sub]
            piece_scale_columns = torch.exp2(
                piece_scales.float() - 127.0
            ).repeat_interleave(32, dim=-1)
            pieces.append((
                wire_bulk[piece_codes.long()].reshape(
                    piece_codes.shape[0], piece_codes.shape[1], -1
                ) * piece_scale_columns
            ).to(torch.bfloat16))
        rebatched = torch.cat(pieces)
        check(f"subbatch{sub}_invariant_{tier_name}",
              torch.equal(rebatched.view(torch.uint16), bulk.view(torch.uint16)))

out = HERE.parent / "receipts/R1_CPU_BITWISE_RECEIPT.json"
out.write_text(json.dumps(
    {"all_pass": all(r["pass"] for r in results), "results": results,
     "torch": torch.__version__, "device": "cpu",
     "note": ("CPU analog of the 8w golden gate: scalar vs bulk dequant chain "
              "bitwise identity + sub-batch invariance. On-host GPU gate is "
              "authoritative for the end-to-end rail.")}, indent=2) + "\n")
print(f"receipt: {out}")
sys.exit(0 if all(r["pass"] for r in results) else 1)
