# Three-tier Luce DDTree N=30 summary

Warm median TG throughput (tok/s), first run dropped.

| corpus | think | AR | +DF | +DDT | +DF/AR | +DDT/AR | +DDT/+DF |
|---|---:|---:|---:|---:|---:|---:|---:|
| sherlock | ON | 10.298 | 33.768 | 55.173 | 3.28x | 5.36x | 1.63x |
| sherlock | OFF | 10.694 | 16.771 | 25.109 | 1.57x | 2.35x | 1.50x |
| codegen | ON | 10.112 | 29.670 | 46.587 | 2.93x | 4.61x | 1.57x |
| codegen | OFF | 10.569 | 20.907 | 32.021 | 1.98x | 3.03x | 1.53x |

AR timings are llama-benchy-reported from patched llama.cpp OpenAI streaming `token_ids`. The old attempted AR package remains invalid as AR and is only used as the +DF/no-DDTree tier.
