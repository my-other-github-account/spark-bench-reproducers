# P968 authority and protocol pin

The genuine-weight arm is task `PUBLIC_TASK` (P967), which is constructing the P943 sealed true-C wire on compute-node-g and serving it through the exact P486 official `DeepseekV4ForCausalLM` vLLM 0.24.0 graph harness. P943 authority is terminal seal SHA-256 `90e6d6b131d14b353be2976848dc90e947cb6fc1cda376e03b760a63dce8d31c` plus active overlay SHA-256 `9a4b709851c62c32f59b17556ef14d53e89cbbfc0fcc93686fc51530e4cf4d62`. The old P486 152/144 row is a harness reference only, not the pending P967 model result.

The only valid IQ4 comparator is task `PUBLIC_TASK`: DeepSeek-V4-Flash UD-IQ4_XS, four exact GGUF shards, compute-node-f head plus compute-node-b RPC donor over QSFP, f16 K/V cache, 17,408 total context, four server slots and four homogeneous client requests. Its preserved row is 161 base / 155 plus-and-both of 164. The older `PUBLIC_TASK` 161/152 receipt is explicitly superseded and must not be mixed into this audit.

Scoring authority is EvalPlus commit `26d6d00bb1fd0fa37f39c99d5290da67891d1c5e`, HumanEvalPlus-v0.1.10 SHA-256 `42526ec0e7d5f3ee0b06d6ced98f8c8bae3d76519151bfb3d36f79010645bd7f`, `min_time_limit=4.0`, `gt_time_limit_factor=4.0`, test details enabled, and network disabled. Raw generation, in-container sanitization, cache removal, and pinned scoring remain distinct stages.

The binding operator amendment uses exactly 5 samples per task, temperature 0.2, top-p 0.95, max 4,096 completion tokens, and paired seed ordinals 10,000–10,004. The greedy-instability arm uses three repetitions at temperature 0, top-p 1, and seed ordinals 20,000–20,002. The prompt is one user message using the exact EvalPlus OpenAIChatDecoder prefix. Canonical task order and four-request grouping are fixed. The same seed ordinal is a matched label, not a claim of common random numbers across vLLM and llama.cpp.

The authority snapshot recorded the TRUE-C runtime as in progress and the exact IQ4 comparator as resource-blocked. Those observations are historical provenance only. Any execution of the amended n=5 protocol must obtain current resource authority rather than relying on this snapshot.
