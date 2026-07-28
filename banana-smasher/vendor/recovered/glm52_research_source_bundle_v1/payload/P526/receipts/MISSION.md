# P526 mission

Host: spark-4 only
Task: task-redacted
Candidate: exactly one explicit-M Triton packed-QTIP/trueVQ gather-dequant-GEMM kernel.
A/B: sealed M=1 decode kernel looped from Python versus candidate at fused13/down, M=128/512.
Gates: finite, >=7.0x each measured row, <=8 GiB transient scratch, no persistent second weight copy.
Forbidden: full product load/serve, services/systemd/tmux/container/WAN/install/delete, any other host.
