# Kernel development and producer state

`runtime/KERNEL_PRODUCERS.json` is the machine-readable producer record for
every active cubin in `runtime/ASSET_MANIFEST.json`. Runtime admission remains
name-, byte-, and SHA-256-exact even when an exact-source-rebuild seal is still
outstanding.

## SM120 W2 and MLA assets

All 26 active SM120 assets are byte-identical to the corresponding outputs in
`Sapid-Labs/vLLM-Moet` commit
`436d2a9100466198fc9cf23bd67a733d87fc9051`. That immutable tree contains the
W2/W4 generators, check harnesses, SASS sources, MLA SASS, and cubins. Its
producer manifest names cubit revision `5912400`, but that short object is not
resolvable from the current upstream and no independent assembler source is
available locally. Therefore these 26 assets are shipped and exactly
hash-gated, but their source reconstruction is not claimed to be sealed.
Recovery of the full assembler identity followed by a byte-identical rebuild
remains an external toolchain gate.

## E43 W3 assets

The six active W3 assets have a complete immutable recipe at
`spark-bench-reproducers` commit
`f252699debc741fa53ed9f569e14ef1951116f21`, path
`deepseek-v4-flash-iq3-vq-warp-gb10/cubins/w3-source`. The recipe pins cubit
`c139df8b34f1dcab607f8ccb685fdea948f3ae4d`, `LUT_LO=0xb6bfc6cd`, and
`LUT_HI=0x4d463c21`, and runs six SASS comparison gates before assembly.

An independent clean extraction and build completed with all six SASS gates
passing and all six generated cubins byte-identical to both the active runtime
assets and the immutable producer receipt. The sealed receipt manifest SHA-256
is `a6effeb493e26e63c56bbec6266063ba8d1e822b39f9d390c8a1d8086c381bf9`.
This producer family is independently rebuilt and byte-identical.

Producer sources remain external immutable references; they are not vendored
into this public runtime repository.
