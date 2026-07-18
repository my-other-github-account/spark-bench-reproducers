# Upstream status at the July 18 research cutoff

The pinned llama.cpp tree had no RPC tensor-range memset implementation. The DSV4 sequence-clear path still called `ggml_backend_tensor_memset`, so a remote tensor could reach an unimplemented backend method.

The tested candidate came from open PR #13601 and adds a client-side RPC backend override that forwards a value-filled byte buffer through the existing tensor-set route. It is small and removed the assertion in the bounded campaign canary, but it is not merged and was previously judged inefficient because the payload scales with the cleared range.

Publication-safe operational conclusions:

1. Build head and RPC binaries from exactly the same commit and shared libraries.
2. Do not treat protocol-patch compatibility as sufficient when serialized operation IDs differ.
3. Require an exact generated-token canary before changing the DSV4 runtime line.
4. Prefer a dedicated validated RPC memset command over transmitting a full byte vector.
5. Keep stable host-KV operation as the fallback until a purpose-built fix passes correctness, reliability, and throughput gates.

Relevant public links:

- <https://github.com/ggml-org/llama.cpp/pull/13601>
- <https://github.com/ggml-org/llama.cpp/issues/25633>
