# MMLU-500 — matched Jul 17 rows

Protocol identity:

- 500 fixed qids; 0-shot choice-loglik;
- question-set SHA256: `24d60b46aa7d0268b5f230760f3caa1391211fdd2893c9073c9e037135b4443a`;
- harness MD5: `db57fd275a727696e4e9bb482958c221`;
- same native non-expert stack for the COMBO-V2 and UD-IQ4_XS routed-expert comparison.

| Artifact | Correct | Accuracy | bpw | Receipt SHA256 |
|---|---:|---:|---:|---|
| Native source reference | 422/500 | 84.4% | 4.49 | `9b07a0e7e36215c87b55a408c6f390ecc88913097041f609f7b28fd10e131d88` |
| UD-IQ4_XS planes | 424/500 | 84.8% | 3.88 | `8b3a8b0c9ec1b798320afa0ae61a98b93aef15094f542867caabb2b4659169c7` |
| COMBO-V2 repaired | 423/500 | 84.6% | 2.887 | `cc8885de0b6ec16c71ab0a2c1abf337fe45f514fd6fda0697df6e5778a84308d` |
| UD-IQ3_XXS | 412/500 | 82.4% | 2.90 | source receipt not yet mirrored publicly |

For COMBO-V2 versus UD-IQ4_XS, the paired gap is −0.2 points, bootstrap 95% CI [−2.2,+1.8], with 13 COMBO-only and 14 UD-only correct answers; exact McNemar p=1.0. The result supports quality-per-bit packaging, not a claim of higher task accuracy.

COMBO-V2's MMLU evaluation was not trained on, but the model inherits an eval-trained warm start. Keep `zero_eval_leakage=false` in every public description.
