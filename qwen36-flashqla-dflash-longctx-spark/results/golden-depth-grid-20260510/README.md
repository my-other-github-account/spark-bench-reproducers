# Corrected Qwen3.6 replacement depth grid

This artifact set supersedes the packaging-only public set at 147b7b2. Final validity includes TG speedup versus paired or proven baselines.

Common shape: TG128, C1, MBT2048, DFlash ON, FlashQLA ON, nspec15, temp0.6/default sampling, no forced top_p/min_tokens/ignore_eos.

| PP | max-model-len | runs | pp mean | tg mean | baseline TG | TG speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 132000 | 3 | 3206.275855798878 | 23.042069823049957 | 11.595354421166252 | 1.9871811577392389 |
| 16384 | 132000 | 3 | 2864.3660874440193 | 31.35508473051363 | 11.082975193390853 | 2.829121619726417 |
| 32768 | 132000 | 3 | 2582.687578720192 | 33.59585002186518 | 24.441134415430362 | 1.3745618125096188 |
| 65536 | 132000 | 3 | 2113.8311627521884 | 30.018358788011078 | 9.664465567931334 | 3.1060547090796318 |
| 131072 | 262144 | 3 | 1553.4603319303994 | 24.150411746982485 | 8.232280473473045 | 2.933623535398555 |
