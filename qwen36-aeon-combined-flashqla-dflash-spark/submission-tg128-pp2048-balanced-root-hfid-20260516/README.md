# LocalMaxxing root-hfId resubmission (2026-05-16)

These payloads resubmit the balanced `pp=2048`, `tg=128`, `c=1`, `N=30` warm-median cells under the parent/root LocalMaxxing model key `Qwen/Qwen3.6-27B` for leaderboard grouping.

Important provenance: the actual served weights remain `AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-NVFP4-MTP-XS` (mounted as `/models/aeon-xs`) with `quantization=NVFP4`; this is stated in each payload's notes. The earlier accepted rows under the exact AEON HF ID are preserved in `../submission-tg128-pp2048-balanced-20260516/localmaxxing_accepted_receipts.json`.

Payloads:
- `localmaxxing-headline-dflash-think-on-code-root-hfid.json` — DFlash / think-on / code headline (`tokSOut=34.94`, `tokSPrefill=2286.10`, `tokSTotal=477.29`, `ttftMs=900.22`).
- `localmaxxing-companion-ar-think-on-code-root-hfid.json` — matched AR/reference companion (`tokSOut=12.04`, `tokSPrefill=2790.38`, `tokSTotal=191.50`, `ttftMs=737.53`).
