# Q2 trainer (Loop B) — root cause + proposed fix T1 (<source-task>)

## Receipt
py-spy 90s @ 50Hz on live s1 trainer PID 2057141 (read-only), speedscope JSON:
/tmp/q2_train_profile.json (s1) + q2_train_profile.json (task workspace).
Step wall 341-383s (~5.8 min at sampling time; card cites 8.4 min/step for the
combo lanes); probe panel 84-104s every 5 steps (~5% amortized).

Top self-time:
  36.5%  autograd backward (_engine_run_backward)
  25.8%  fullmenu_surface.py:259  payload[codes][rows].to("cuda")
   9.2%  fullmenu_surface.py:260  payload[sc][rows].to("cuda")
   6.6%  torch grad_mode ctx init
   2.9%  lp4_train.evict_tensor

## Root cause (code-level, fullmenu_surface.py)
Lines 259/260 are NOT dequant compute — they are the mmap ROW GATHER of frozen
codes/scales plus H2D. Three multipliers make this ~35% of the step:
  1. _evict_payloads() (line 317) runs after EVERY surface forward and
     madvise(DONTNEED)s the payload mmaps — deliberately, to protect
     unified-memory cudaMalloc. Consequence: the next forward re-faults the
     same pages from NVMe.
  2. Gradient checkpointing re-executes each layer forward in backward
     (binrepair_e2e.batch_loss), re-gathering the same rows again.
  3. Grad accumulation (BR_BATCH=4) repeats 1+2 four times per step.
  => the same frozen bytes are re-read from disk ~8x per step per layer.

## Key fact enabling the fix
Codes and scales are FROZEN in this mechanism (r5-fullmenu-combo-v1 trains
codebooks + rmsnorms + attention output gains only; n_trainable=7,141,035 =
6,694,912 cb + 446,080 norm + 43 gains). The gathered (codes, scales) GPU
tensors per (layer, tier, projection, row-chunk) are constant for the entire
run. Only the codebook input to BatchedGenericVQDeqFn changes between steps.

## Proposed T1 patch (trainer-side, ~20 lines)
In _weights_for(): key a GPU-side cache on (L, projection, source, tier,
tuple(rows-chunk)) holding {codes_gpu, sc_gpu} u8/i16 tensors; on hit, skip the
mmap gather + H2D and call BatchedGenericVQDeqFn.apply(codebook, codes_gpu,
sc_gpu) directly. Keep _evict_payloads() as-is (it protects the mmaps; the
cache removes the need to re-fault them). Autograd path is untouched — the
Function sees identical inputs, codebook grads flow exactly as before.
VRAM cost: codes+scales for hit experts across 43 layers x 2 proj — bounded by
the pack's code bytes for hit experts (~14 GiB worst-case all-256-hit at d8;
gate behind R5_CODES_CACHE=1 env with LRU cap R5_CODES_CACHE_GIB, default 8,
evicting whole layers LRU; s1 trainer holds ~113 GiB free unified mem today).

## Expected effect
Removes ~35% of step self-time minus one cold gather per (layer,chunk) per run
=> step 341-383s -> ~230-260s (~1.45x). Backward (36.5%) untouched.
NOT claimed as the 3x path — the rail loop (Loop A) is; this is the ranked
train lever with receipts for a follow-up card.

## Bit-exactness
Same tensors, same op order, same dtypes — cache-hit tensors are the exact
device copies the sealed path would have created. Validation: fixed-seed short
run, assert per-step loss bitwise equality with cache on/off before adoption.
