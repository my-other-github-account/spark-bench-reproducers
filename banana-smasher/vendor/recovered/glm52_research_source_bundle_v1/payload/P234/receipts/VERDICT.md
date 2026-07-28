# Clean HE164 Transfer-8 EvalPlus Verdict

- Checkpoint SHA-256: `4086e9d8be9ece067ce3b713c22654e59bcad614af9444bdfacd2e66e0a02fd5`
- Dataset: HumanEvalPlus-v0.1.10 (`42526ec0e7d5f3ee0b06d6ced98f8c8bae3d76519151bfb3d36f79010645bd7f`)
- EvalPlus commit/image: `26d6d00` / `sha256:ce82d4f2e99754feb576991dec8d558096cbcb43644b53faf941324d77981c95`
- Raw generations sealed: **164/164**
- HumanEval base pass@1: **161/164**
- HumanEval+ pass@1: **152/164**
- Both pass@1: **152/164**
- Binding clean claim (HumanEval base >=160/164): **PASS_GE_160_OF_164**
- Delta vs sealed STEP0 161/164: **+0**
- Delta vs UD-IQ4_XS 161/164 reference: **+0**
- Generation: temperature=0, top_p=1.0, max_tokens=4096, seed=0, fingerprint=`vllm-0.24.0-3f34bf12`

Failure sets and all per-task outputs are in `VERDICT.json`.
