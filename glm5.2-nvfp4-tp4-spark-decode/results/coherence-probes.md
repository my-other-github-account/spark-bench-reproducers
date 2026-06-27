# Coherence probes — served GLM-5.2-NVFP4 TP=4 (dense, index_topk=0)

All probes against the live serve, `temperature=0` (deterministic), `max_tokens` 50–90.
The point: confirm the **dense** path (DSA sparse indexer OFF via `index_topk=0`) is not degraded —
output is correct across arithmetic, multi-step reasoning, formal logic, and recall.

| Prompt | Model output (verbatim, trimmed) | Verdict |
|---|---|---|
| Is 2+2=5? | "No, 2+2=4. This is a basic mathematical fact that is universally accepted and proven. The equation 2+2=5 is incorrect…" | ✅ correct |
| What is the capital of France? | "The capital of France is Paris. … Situated in the north-central part of the country along the Seine River…" | ✅ correct |
| 17 × 23? | "391 … The answer is 391. Here is how to calculate 17 x 23 using the long multiplication method…" | ✅ correct |
| Train 60 mi in 1.5 h, avg mph? | "40 mph. … divide the total distance by the total time. 60 miles / 1.5 hours = 40 mph." | ✅ correct |
| All cats are mammals; Fluffy is a cat; is Fluffy a mammal? | "Yes, Fluffy is a mammal because all cats are mammals, and Fluffy is a cat. This is a valid deductive argument using the rule of universal instantiation." | ✅ correct (named the inference rule) |
| Capital of France, then "hello" in French | "Paris, Bonjour. … Bonjour is derived from the French words bon, meaning good, and jour, meaning day." | ✅ correct |
| Explain a transformer neural network | "…a type of deep learning model designed to process sequential data … using self-attention mechanisms to weigh the importance of different words … process entire sequences simultaneously, allowing for parallelization … encoder-decoder architecture …" | ✅ coherent, accurate |

## On dense vs. sparse (why this is not a degraded model)

`index_topk=0` disables the DSA "lightning indexer" and runs **dense MLA** — full attention over all
tokens. Dense is the *exact* attention computation; the sparse indexer is an *approximation* that
prunes the KV cache to save compute/memory at long context. Turning it off removes an **efficiency**
feature, not a **capability** feature — by construction dense is the quality ceiling, not a regression.

Caveat: the sparse-vs-dense difference only manifests at **long context** (thousands of tokens), where
the indexer would actually prune. This serve is capped at `--max-model-len 256`, so the long-context
regime is out of scope here; all probes are within the coherent short-context window. (On GLM-5.1 the
DSA indexer was load-bearing and dense produced token-salad; GLM-5.2, like GLM-5.0, is coherent dense —
which is exactly what makes this dense config valid for 5.2.)
