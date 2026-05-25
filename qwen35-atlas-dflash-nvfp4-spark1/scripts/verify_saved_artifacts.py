#!/usr/bin/env python3
import json, pathlib, sys
root=pathlib.Path(__file__).resolve().parents[1]
summary=json.loads((root/'results/nvfp4_final_summary_legal_21tps.json').read_text())
errors=[]
def req(cond,msg):
    if not cond: errors.append(msg)
req(summary.get('model_format')=='nvfp4','model_format must be nvfp4')
req(summary.get('fair_baseline') is True,'fair_baseline must be true')
req(summary.get('quality_spotcheck_pass') is True,'quality_spotcheck_pass must be true')
req(summary.get('prompt_count',0)>=72,'prompt_count must be >=72')
req(summary.get('concurrency')==1,'concurrency must be 1')
req(summary.get('temperature')==0.0,'temperature must be 0.0')
req(summary.get('max_tokens')==160,'max_tokens must be 160')
req(summary.get('dflash_output_tps',0) >= 21.28429140169363 - 1e-12,'retained legal DFlash TPS mismatch')
req(summary.get('speedup',0) >= 1.5,'speedup must be >=1.5x')
for rel in ['results/ar_benchmark_q35_nvfp4_full72_c1.json','results/dflash_benchmark_q35_nvfp4_full72_c1_gamma3_all.json','prompts/atlas_diverse_72.jsonl','patches/0001-atlas-nvfp4-dflash-all-quant-pass-state.patch']:
    req((root/rel).exists(), f'missing {rel}')
if errors:
    print('FAIL saved-artifact verification')
    print('\n'.join('- '+e for e in errors))
    sys.exit(1)
print('PASS saved-artifact verification')
print(json.dumps({k: summary[k] for k in ['prompt_count','ar_output_tps','dflash_output_tps','speedup','model_format','fair_baseline','quality_spotcheck_pass']}, indent=2))
