# Environment capture

Reproduction requires both package pins and platform evidence. Environment snapshots are grouped by role because build, rail, repair, and serve hosts evolved independently during the campaign.

Each snapshot contains:

- `python --version` and a sorted `pip freeze`;
- CUDA toolkit and `nvcc` versions;
- NVIDIA driver version and GPU identity;
- kernel, architecture, and OS release;
- vLLM Git commit plus local patch manifest;
- source-checkpoint revision and relevant artifact hashes.

No host addresses, usernames, home directories, credentials, caches, or model weights belong here. Public commands use `$MISSION_ROOT`, `$MODEL_ROOT`, and `spark-N` aliases.

GB10 note: system RAM and GPU allocations share a unified pool. Reproduction logs must report `MemAvailable`, active GPU allocations, and whether expert planes are anonymous-resident or file-backed. Lowering a nominal GPU utilization fraction only shrinks KV cache; it does not create physical memory.
