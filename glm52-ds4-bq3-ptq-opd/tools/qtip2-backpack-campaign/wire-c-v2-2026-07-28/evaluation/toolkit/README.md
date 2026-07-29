# P968 evaluation toolkit

This standard-library orchestration layer implements the preregistered paired TRUE-C/IQ4 EvalPlus protocol. It publishes protocol and runnable tooling only; this package contains no completed paired functional result.

Binding arms:

- sampled: exactly 5 completions per task, temperature 0.2, top-p 0.95, seeds 10000–10004;
- greedy instability: 3 repeats per task, temperature 0, top-p 1, seeds 20000–20002;
- max completion tokens: 4096;
- canonical dataset order with four concurrent task requests.

The TRUE-C runner takes its deployment locations from `P968_TRUE_C_SOURCE_ROOT`, `P968_TRUE_C_MODEL_ROOT`, `P968_HOST_CLAIM`, and `P968_TRUE_C_HOSTNAME`. Public placeholders are intentionally non-operational until an authorized operator binds them to SHA-verified inputs.

Run the local protocol tests:

```bash
python3 -m unittest discover -s evaluation/toolkit/tests -v
```

Actual generation/scoring remains gated by current resource authority, exact model/runtime hashes, full declared coverage, separately hashed raw/sanitized/scored outputs, and the publication rules in `../../EVALUATION_PROTOCOL.md`.
