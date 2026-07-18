# GB10 3003 MHz no-reboot clock lock

## Recipe

```bash
sudo nvidia-smi -lgc 3003,3003
nvidia-smi --query-gpu=clocks.sm,power.draw --format=csv -l 1
```

The lock applied without reboot. Verify under real decode load; an idle readback is insufficient.

## Fleet audit

Host identities are anonymized, but the fleet-level receipt is preserved:

| Audit population | Count | Observed state | Action |
|---|---:|---|---|
| Reachable GB10 nodes | 7 | 2437–2541 MHz SM versus 3003 MHz maximum (81–85%) | `-lgc 3003,3003` accepted on all seven |
| Offline node | 1 | not measured | apply and verify on recovery |
| Reachable CPU governors | 7 | `performance` | no change |

One detailed session receipt records a 2496 MHz pre-lock clock. The fleet also showed accumulated software-power-cap time, so a successful setter return is not sufficient evidence of a sustained 3003 MHz workload clock.

To reset:

```bash
sudo nvidia-smi -rgc
```

## Measured session

The same service session reported:

| Concurrency | Throughput |
|---:|---:|
| 1 | 12.571 tok/s |
| 2 | 19.625 aggregate tok/s |
| 4 | 29.257 aggregate tok/s |

The throughput receipt belongs to the IQ4_XS RPC benchmark and is cross-linked in `../iq4-rpc-memset/`. The clock lock is a configuration receipt, not proof that all throughput gain came from clocks.

Label every row `pre-lock` or `post-lock`. Never mix both states inside one five-row median. A post-lock row must retain a 1 Hz `clocks.sm,power.draw` trace during decode and report any sustained sag below roughly 2.9 GHz.

## Persistence

Reapply and verify after reboot, driver reload/reset, or power-state transition. Monitor power and sustained SM clock while the actual workload runs.
