# V4 wire production pipeline and `smash export` parity

Task: `t_61728a75`
Basis: U012 source model index SHA-256 `58c9d59dfe8fd1e7e833be131043f4b45bfa27064fc19b9fa4fffa6475f2d0fc`
Assignment: `ASSIGNMENT_WIRE_C_V2`, SHA-256 `f521cf07e0dce3c39739c7493b6eda82cd78d6b1566fadb2101691321566ca39`

## Behavioral reference

The reference producer is:

- `deepseek-v4-flash-iq3-vq-warp-gb10/tools/prepack_iq3_vq.py`
- `deepseek-v4-flash-iq3-vq-warp-gb10/tools/iq3_vq_wire.py`

It produced the `iq3-vq-wire-v1` layer format and the `wire_v4-step32` serve tree. The code is a behavioral reference only; `smash export` does not invoke, import, paste, or subprocess it.

## Ordered V4 production pipeline

1. Normalize the solver assignment for all 43 layers, 256 experts, and both projections (`fused13`, `down`). Reject missing, duplicate, or unsupported cells.
2. Resolve each assigned tier from the selected source planes. For every layer/projection, gather exactly the expert rows named by the assignment; do not carry unselected tier payloads into the wire pack.
3. Bind repaired U012 state. Repair may replace D4 codebooks and serving-side norm/output payloads, but it must not rewrite code indices or the assignment.
4. Group selected VQ rows by tier. Validate D4 codebook dimensions and power-of-two cardinality; derive index width as `log2(k)` (10, 11, or 12 bits for k=1024/2048/4096).
5. Pack every D4 output row independently in little-endian, least-significant-bit-first order. The reference operation is equivalent to `np.packbits(index_bits, axis=-1, bitorder="little")`; row size is `ceil(values_per_row * index_bits / 8)`. Row boundaries are preserved so a runtime kernel can address a row without scanning earlier rows.
6. Preserve scale bytes verbatim. Emit repaired codebooks and the addressing metadata needed by the decoder: per-expert code offset, scale offset, packed row bytes, vector dimension, index width, and codebook offset.
7. Preserve QTIP trellis payloads as their existing packed 16-bit wire words. Their apparent `int16` dtype is a container representation, not unpacked 2/3-bit indices.
8. Write each layer payload atomically, then write layer metadata containing exact file byte counts and SHA-256 hashes.
9. Write the pack manifest, its SHA-256 sidecar, and `PACK_COMPLETE` only after all 43 layer records are complete.
10. Run the pack validator and compare per-layer/per-file-class logical bytes against `wire_v4-step32` before serving.

## Diff against the pre-fix `smash export` pipeline

| Stage | V4 reference | Pre-fix `smash export` | Corrected native stage |
|---|---|---|---|
| Selection ownership | Assignment chooses every expert/projection row | Export admitted the full P1016 payload tree | Manifest-owned tier/slot/family bindings select only referenced payloads and reject missing, duplicate, or family-drifted cells |
| D4 code storage | Independent LSB-first rows at 10/11/12 bits | Selected D4 `codes.npy` remained dense `int16` hardlinks | `_pack_index_npy` writes canonical `uint8` row-packed NPY payloads during export |
| Decode metadata | Width/row bytes/offsets are explicit | Dense shape/dtype implied direct indexing | Manifest records encoding, index width, values and bytes per row, decoded dtype/shape/bytes/SHA-256 |
| QTIP handling | Existing packed words preserved | Hardlinked | Still hardlinked; no attempted 2/3-bit reinterpretation |
| Repair boundary | Codebooks/norms/output gains may change; indices do not | Repair materialization existed but was not paired with physical D4 packing | Repair remains architecture-owned and fails if it attempts to rewrite packed D4 indices |
| Runtime | Kernel decodes packed indices | Loader/kernel expected dense D4 codes | Loader requires packed metadata and kernel decodes 1–16-bit little-endian rows |
| Validation | File hashes plus wire structure | File hashes only proved the dense export | Validator reconciles NPY headers, packed/decoded metadata, selected family bindings, hashes, and resident-byte accounting |

## Canonical `smash export` stage after correction

For `p1016-true-c-native-planes-v1`, the export verb now performs selection verification, repair binding, D4 row packing, non-D4 zero-copy hardlinking, metadata/base-weight materialization, manifest creation, and self-verification in one architecture-owned call. It does not launch an old script or invent a second pack command. The producer label is `smash export:v4-row-packed-selected-wire-v1`.

The full U012 action must use the fixed verb with the sealed native-plane source, `ASSIGNMENT_WIRE_C_V2`, U012 repair checkpoint, active overlay, explicit runtime-floor receipt, and same-filesystem hardlink mode. Final acceptance is `validate-pack` PASS plus a per-layer byte-parity receipt against `wire_v4-step32`.
