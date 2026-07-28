#!${SPARK_HOME}/humming_env/bin/python
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import threading
import time

import torch

MISSION = Path('${SPARK_HOME}/missions/P526_QTIP_MBATCH_t_88eede57_s4')
PACKED = Path('${SPARK_HOME}/missions/QTIP_SERVE_C2_t_91ac9ee9_s4/packed/vq8u_layer_000.pt')
SEALED = MISSION / 'code/qtip_vq_backend_sealed.py'
CANDIDATE = MISSION / 'code/mbatched_qtip.py'
PROGRESS = MISSION / 'PROGRESS.json'
RESULT = MISSION / 'RESULT.json'


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def status_kb(name: str) -> int:
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith(name + ':'):
            return int(line.split()[1])
    return 0


def mem_available_kb() -> int:
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1])
    return 0


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def atomic_json(path: Path, obj: dict):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')
    os.replace(tmp, path)


def elapsed_ms(fn, reps: int) -> tuple[float, torch.Tensor, list[float]]:
    vals = []
    out = None
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        end.synchronize()
        vals.append(float(start.elapsed_time(end)))
    return statistics.median(vals), out, vals


def main():
    MISSION.mkdir(parents=True, exist_ok=True)
    (MISSION / 'run').mkdir(exist_ok=True)
    (MISSION / 'run/bench.pid').write_text(str(os.getpid()) + '\n')
    started = time.time()
    progress = {
        'schema': 'p526-progress-v1', 'task_id': 'task-redacted', 'host': 'spark-4',
        'phase': 'STARTED_REAL_GPU_MICROBENCH', 'pid': os.getpid(), 'started_unix': started,
        'completed_shapes': [], 'candidate_kernel_count': 1,
    }
    atomic_json(PROGRESS, progress)
    sealed = load_module('qtip_vq_backend_sealed', SEALED)
    cand = load_module('mbatched_qtip', CANDIDATE)
    torch.manual_seed(526)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')

    stop = threading.Event()
    nadir = {'kb': mem_available_kb()}
    def sample_mem():
        while not stop.wait(0.05):
            nadir['kb'] = min(nadir['kb'], mem_available_kb())
    sampler = threading.Thread(target=sample_mem, daemon=True)
    sampler.start()

    data = torch.load(PACKED, map_location='cpu', mmap=True, weights_only=True)
    rows = []
    definitions = [
        ('fused13', 'codes13', 'sc13', 'cb13', 4096, 4096),
        ('down', 'codes2', 'sc2', 'cb2', 4096, 2048),
    ]
    try:
        for proj, ck, sk, bk, n, k in definitions:
            codes = data[ck][0:1].to('cuda').contiguous()
            scales = data[sk][0:1].to('cuda').contiguous()
            cb = data[bk].to('cuda').contiguous()
            expert_ids = torch.zeros((1,), device='cuda', dtype=torch.int32)
            projection = sealed.PackedProjection(codes, scales, cb, n, k)
            resident_packed_bytes = projection.resident_bytes
            for m in (128, 512):
                x = torch.randn((1, m, k), device='cuda', dtype=torch.bfloat16)
                def incumbent():
                    return torch.cat([projection.forward(x[:, i:i+1, :], expert_ids) for i in range(m)], dim=1)
                def candidate():
                    return cand.qtip_gemm_mbatched(x, codes, scales, cb, expert_ids)

                # Real cold compile/warmup of both paths, excluded from timing.
                y_cand = candidate()
                y_inc = incumbent()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                alloc0 = torch.cuda.memory_allocated()
                cand_ms, y_cand, cand_trials = elapsed_ms(candidate, 5)
                cand_peak = torch.cuda.max_memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                inc_alloc0 = torch.cuda.memory_allocated()
                inc_ms, y_inc, inc_trials = elapsed_ms(incumbent, 3)
                inc_peak = torch.cuda.max_memory_allocated()

                a = y_inc.float()
                b = y_cand.float()
                abs_err = (a - b).abs()
                rel_err = abs_err / a.abs().clamp_min(1e-5)
                finite = bool(torch.isfinite(b).all().item())
                row = {
                    'projection': proj, 'R': 1, 'M': m, 'N': n, 'K': k,
                    'incumbent_kind': 'sealed M=1 _vq_gemv looped from Python',
                    'candidate_kind': 'one explicit-M Triton gather-dequant-GEMM launch grid',
                    'incumbent_wall_ms_median': inc_ms,
                    'incumbent_wall_ms_trials': inc_trials,
                    'candidate_wall_ms_median': cand_ms,
                    'candidate_wall_ms_trials': cand_trials,
                    'speedup': inc_ms / cand_ms,
                    'incumbent_tokens_per_s': m * 1000.0 / inc_ms,
                    'candidate_tokens_per_s': m * 1000.0 / cand_ms,
                    'finite': finite,
                    'output_abs_mean': float(b.abs().mean().item()),
                    'max_abs_error': float(abs_err.max().item()),
                    'mean_abs_error': float(abs_err.mean().item()),
                    'max_relative_error': float(rel_err.max().item()),
                    'mean_relative_error': float(rel_err.mean().item()),
                    'packed_resident_bytes': resident_packed_bytes,
                    'candidate_peak_scratch_bytes': int(max(0, cand_peak - alloc0)),
                    'incumbent_peak_scratch_bytes': int(max(0, inc_peak - inc_alloc0)),
                    'no_persistent_second_weight_copy': True,
                }
                rows.append(row)
                progress.update({
                    'phase': 'RUNNING', 'completed_shapes': [f"{r['projection']}:M{r['M']}" for r in rows],
                    'latest': row, 'mem_available_nadir_kb': nadir['kb'], 'vmhwm_kb': status_kb('VmHWM'),
                })
                atomic_json(PROGRESS, progress)
                print(json.dumps(row, sort_keys=True), flush=True)
                del x, y_cand, y_inc, a, b, abs_err, rel_err
                torch.cuda.empty_cache()
            del codes, scales, cb, expert_ids, projection
            torch.cuda.empty_cache()
    finally:
        stop.set(); sampler.join(timeout=1)

    min_speedup = min(r['speedup'] for r in rows)
    all_finite = all(r['finite'] for r in rows)
    max_scratch = max(r['candidate_peak_scratch_bytes'] for r in rows)
    verdict = 'PASS' if all_finite and min_speedup >= 7.0 and max_scratch <= 8 * 1024**3 else 'MISS'
    result = {
        'schema': 'p526-qtip-mbatched-ab-v1', 'task_id': 'task-redacted', 'host': 'spark-4',
        'verdict': verdict, 'gate': 'all finite, min speedup >=7.0x, candidate scratch <=8GiB',
        'rows': rows, 'min_speedup': min_speedup, 'all_finite': all_finite,
        'max_candidate_peak_scratch_bytes': max_scratch,
        'vmhwm_kb': status_kb('VmHWM'), 'mem_available_nadir_kb': nadir['kb'],
        'candidate_sha256': sha256(CANDIDATE), 'incumbent_sha256': sha256(SEALED),
        'packed_sample_path': str(PACKED), 'packed_sample_stat_bytes': PACKED.stat().st_size,
        'candidate_kernel_count': 1, 'persistent_second_weight_copy': False,
        'started_unix': started, 'completed_unix': time.time(), 'pid': os.getpid(),
    }
    atomic_json(RESULT, result)
    progress.update({'phase': 'COMPLETE', 'verdict': verdict, 'result': str(RESULT)})
    atomic_json(PROGRESS, progress)
    (MISSION / 'DONE').write_text(verdict + '\n')
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
