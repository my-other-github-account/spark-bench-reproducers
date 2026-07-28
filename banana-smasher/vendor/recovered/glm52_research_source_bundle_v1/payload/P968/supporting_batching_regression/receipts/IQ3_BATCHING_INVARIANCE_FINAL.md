# IQ3 10-row batching-invariance gate

Status: PASS
Scope: diagnostic only; no new table row and no full rerun authorized. Protected clean IQ4 161/155/155 was not mutated.
Model: UD-IQ3_XXS (102,999,887,616 bytes; exact four-shard receipt fd71923d010923a285dbd7168730ffdb6e72c296bdf604369481b80311e3422d)
Contract: f16 K/V, ctx=17408, temp=0, top_p=1, seed=0, max=4096; A server/client=1/1; B=4/4.
Scoring: pinned EvalPlus 26d6d00 on spark-2, Docker network none.

Visible/token batching invariant: False
Generation differing rows: ['HumanEval/32', 'HumanEval/39', 'HumanEval/49', 'HumanEval/76', 'HumanEval/91', 'HumanEval/99', 'HumanEval/116', 'HumanEval/127', 'HumanEval/129', 'HumanEval/145']
Base/plus outcome invariant: False
Outcome differing rows: ['HumanEval/39', 'HumanEval/76']
Selected-arm counts A: {'base': 7, 'both': 5, 'denominator': 10, 'plus': 5}
Selected-arm counts B: {'base': 8, 'both': 4, 'denominator': 10, 'plus': 4}

## Per row

| Row | Visible equal | Finish/tokens equal | A base/plus | B base/plus | A wall s | B wall s |
|---|---:|---:|---|---|---:|---:|
| HumanEval/32 | False | False | fail/fail | fail/fail | 162.392 | 373.270 |
| HumanEval/39 | False | False | pass/pass | pass/fail | 51.092 | 153.196 |
| HumanEval/49 | False | False | pass/pass | pass/pass | 46.136 | 102.757 |
| HumanEval/76 | False | False | fail/fail | pass/fail | 312.358 | 286.612 |
| HumanEval/91 | False | False | pass/fail | pass/fail | 89.199 | 108.608 |
| HumanEval/99 | False | False | pass/fail | pass/fail | 63.710 | 107.414 |
| HumanEval/116 | False | False | pass/pass | pass/pass | 300.887 | 705.221 |
| HumanEval/127 | False | False | pass/pass | pass/pass | 93.999 | 118.328 |
| HumanEval/129 | False | False | pass/pass | pass/pass | 213.913 | 648.235 |
| HumanEval/145 | False | False | fail/fail | fail/fail | 102.530 | 538.392 |

## Cleanup

spark-4 exact-released UNCLAIMED, GPU-empty, task-process-empty. Released claim SHA256: 1ab99860ca2be3283bb55f043ef9ea274b3215b3305454864bd74a2a5020cd9b
spark-4 disk free at release: 43,245,039,616 bytes.
spark-2 scoring host exact-released UNCLAIMED, GPU-empty, task-process-empty. Released claim SHA256: 648d927223b512e6e975291d75bc3f2625068f2440553266ea8c8ed314c92a2c
spark-2 disk free at release: 102,896,459,776 bytes.
