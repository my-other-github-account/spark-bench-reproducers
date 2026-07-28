#!/usr/bin/env python3
import ast
import hashlib
import json
import os
import pathlib
import re
import subprocess
import tempfile
import time

import torch

ROOT = pathlib.Path('${SPARK_HOME}/missions/P959_TRUE_C_REPAIR_t_8343707a_s3')
TASK = 'task-redacted'
CLAIM_SHA = 'e095527068f0474eb8ed6e93b0038643b8d1409cdd1b2a58e109df302334db48'
OLD_SEED_SHA = '0bd644edc542460bf950b576e23480d89748398aab00831e3ea47a27933efca1'
OLD_REBUILD_SHA = '94ebeadb1275710daaf0f8da9b793a2dd69e742a692709cdb8dc666656a905c5'
LAUNCHER_SHA = 'a20c0b645937729aafd1fbf97758941a40162585c129367d91e0575197e17c7e'
TRAINER_SHA = '36a7e9d0a8be3fa6b1d2612a6a58c9032667aab56c7f20e6f50e5547b78c6921'
TERMINAL_SHA = '90e6d6b131d14b353be2976848dc90e947cb6fc1cda376e03b760a63dce8d31c'
OVERLAY_SHA = '9a4b709851c62c32f59b17556ef14d53e89cbbfc0fcc93686fc51530e4cf4d62'
KEY_RE = re.compile(r'^f521_L(?P<layer>\d{3})_(?P<proj>down|fused13)_d(?P<dim>\d+)_k(?P<k>\d+)_(?P<sha>[0-9a-f]{64})$')


def sha(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_bytes(path, data):
    path = pathlib.Path(path)
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        dfd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_json(path, obj):
    atomic_bytes(path, (json.dumps(obj, indent=2, sort_keys=True) + '\n').encode())


def atomic_torch(path, obj):
    path = pathlib.Path(path)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    try:
        torch.save(obj, tmp)
        with tmp.open('rb') as f:
            os.fsync(f.fileno())
        os.replace(tmp, path)
        dfd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        tmp.unlink(missing_ok=True)


def no_scoped_processes():
    needles = (
        str(ROOT / 'code/run_p959_controller.sh'),
        str(ROOT / 'code/run_p959_terminal_repair.sh'),
        str(ROOT / 'code/genesis_basic_repair.py'),
        str(ROOT / 'code/p911_resource_guard.py'),
    )
    rows = []
    for p in pathlib.Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name) in {os.getpid(), os.getppid()}:
            continue
        try:
            cmd = (p / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
        except Exception:
            continue
        if any(n in cmd for n in needles):
            rows.append({'pid': int(p.name), 'cmdline': cmd})
    return rows


def main():
    claim_path = pathlib.Path('${SPARK_HOME}/HOST_CLAIM.json')
    claim = json.loads(claim_path.read_text())
    if sha(claim_path) != CLAIM_SHA or claim.get('owner') != TASK or claim.get('mission') != str(ROOT):
        raise RuntimeError('exact P959 claim drift')
    gpu = subprocess.run(
        ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    scoped = no_scoped_processes()
    if gpu or scoped:
        raise RuntimeError(f'host not empty: gpu={gpu!r} scoped={scoped}')
    if any((ROOT / 'checkpoints').glob('*')):
        raise RuntimeError('no-update completion requires empty checkpoints directory')
    root_free = os.statvfs('/').f_bavail * os.statvfs('/').f_frsize
    if root_free < 8 * 1024**3:
        raise RuntimeError(f'root floor failed: {root_free}')
    if sha(ROOT / 'code/run_p959_terminal_repair.sh') != LAUNCHER_SHA:
        raise RuntimeError('launcher drift')
    if sha(ROOT / 'code/genesis_basic_repair.py') != TRAINER_SHA:
        raise RuntimeError('trainer drift')
    terminal_path = ROOT / 'inputs/P943_TRUE_C_TERMINAL_SEAL.json'
    terminal = json.loads(terminal_path.read_text())
    if sha(terminal_path) != TERMINAL_SHA or terminal.get('active_overlay_sha256') != OVERLAY_SHA:
        raise RuntimeError('P943 terminal authority drift')

    status_path = ROOT / 'run/BASIC_REPAIR_STATUS.json'
    status_sha = sha(status_path)
    status = json.loads(status_path.read_text())
    prefix = 'RuntimeError: terminal seed extension requires explicit gate: '
    error = status.get('error', '')
    if not error.startswith(prefix):
        raise RuntimeError(f'expected exact strict-subset failure, got {error!r}')
    extras_by_layer = ast.literal_eval(error[len(prefix):])
    extras = [(label, key) for label, keys in sorted(extras_by_layer.items()) for key in keys]
    if len(extras) != 12 or {label for label, _ in extras} != {'L0', 'L1', 'L2'}:
        raise RuntimeError(f'unexpected live-only inventory: {extras_by_layer}')
    if any(len(extras_by_layer[label]) != 4 for label in ('L0', 'L1', 'L2')):
        raise RuntimeError(f'unexpected per-layer live-only inventory: {extras_by_layer}')

    seed_path = ROOT / 'seed/TERMINAL_UPDATE_000.pt'
    rebuild_path = ROOT / 'receipts/P959_TERMINAL_UPDATE_000_SEED_REBUILD.json'
    if sha(seed_path) != OLD_SEED_SHA or sha(rebuild_path) != OLD_REBUILD_SHA:
        raise RuntimeError('184-key terminal seed/rebuild receipt drift')
    seed = torch.load(seed_path, map_location='cpu', mmap=True, weights_only=False)
    codebooks = seed['state']['codebooks']
    before_count = sum(len(v) for v in codebooks.values())
    if before_count != 184:
        raise RuntimeError(f'old terminal seed count drift: {before_count}')

    package = pathlib.Path('${SPARK_HOME}/missions/P875_QTIP3_FORTRESS_t_67604030_s3/base_wire')
    bindings = []
    for label, key in extras:
        match = KEY_RE.fullmatch(key)
        if not match or label != f"L{int(match.group('layer'))}":
            raise RuntimeError(f'malformed live-only key: {label}/{key}')
        layer = int(match.group('layer'))
        proj = match.group('proj')
        dim = int(match.group('dim'))
        k = int(match.group('k'))
        expected_sha = match.group('sha')
        source = package / f'layer_{layer:03d}' / f'd{dim}_k{k}.{proj}.codebook.fp16.bin'
        if not source.is_file() or source.stat().st_size != dim * k * 2 or sha(source) != expected_sha:
            raise RuntimeError(f'physical codebook binding drift: {source}')
        if key in codebooks[label]:
            raise RuntimeError(f'live-only key unexpectedly already in seed: {label}/{key}')
        raw = source.read_bytes()
        master = torch.frombuffer(bytearray(raw), dtype=torch.float16).clone().reshape(k, dim).float()
        roundtrip = master.to(torch.float16).contiguous().numpy().tobytes()
        if hashlib.sha256(roundtrip).hexdigest() != expected_sha:
            raise RuntimeError(f'fp32 master round-trip drift: {label}/{key}')
        codebooks[label][key] = master
        bindings.append({
            'layer': layer,
            'key': key,
            'source': str(source),
            'bytes': len(raw),
            'wire_fp16_sha256': expected_sha,
            'master_fp32_sha256': hashlib.sha256(master.contiguous().numpy().tobytes()).hexdigest(),
            'shape': [k, dim],
            'reason': 'actual live physical base-wire codebook retained alongside P943 replacement group',
        })

    after_count = sum(len(v) for v in codebooks.values())
    if after_count != 196:
        raise RuntimeError(f'complete terminal seed count drift: {after_count}')
    seed.setdefault('identity', {})['p959_terminal_actual_live_inventory'] = {
        'p943_terminal_sha256': TERMINAL_SHA,
        'active_overlay_sha256': OVERLAY_SHA,
        'previous_terminal_seed_sha256': OLD_SEED_SHA,
        'previous_terminal_codebooks': before_count,
        'completed_terminal_codebooks': after_count,
        'physical_base_live_only_added': len(bindings),
        'old_to_new_alias_mapping_used': False,
        'speculative_seed_used': False,
        'training_updates_run': 0,
        'strict_subset_failure_status_sha256': status_sha,
        'bindings': bindings,
    }
    seed['saved_unix'] = time.time()
    seed['host'] = 'spark-3'

    preserved = ROOT / 'seed/TERMINAL_UPDATE_000_184KEY_PREIMAGE.pt'
    if preserved.exists():
        if sha(preserved) != OLD_SEED_SHA:
            raise RuntimeError('preserved 184-key seed preimage drift')
    else:
        os.link(seed_path, preserved)
        if sha(preserved) != OLD_SEED_SHA:
            raise RuntimeError('failed to preserve 184-key seed preimage')
    atomic_torch(seed_path, seed)
    new_seed_sha = sha(seed_path)
    check = torch.load(seed_path, map_location='cpu', mmap=True, weights_only=False)
    if sum(len(v) for v in check['state']['codebooks'].values()) != 196:
        raise RuntimeError('new seed readback count drift')
    for binding in bindings:
        tensor = check['state']['codebooks'][f"L{binding['layer']}"][binding['key']]
        if hashlib.sha256(tensor.to(torch.float16).contiguous().numpy().tobytes()).hexdigest() != binding['wire_fp16_sha256']:
            raise RuntimeError(f"new seed readback byte drift: {binding['key']}")

    old_rebuild = json.loads(rebuild_path.read_text())
    rebuild = dict(old_rebuild)
    rebuild.update({
        'status': 'PASS_TERMINAL_UPDATE_000_REBUILT_FROM_WIRE',
        'completion_status': 'PASS_COMPLETE_ACTUAL_LIVE_KEYSPACE_NO_EXTENSION_GATE',
        'previous_rebuild_receipt_sha256': OLD_REBUILD_SHA,
        'previous_terminal_seed': str(preserved),
        'previous_terminal_seed_sha256': OLD_SEED_SHA,
        'terminal_seed_sha256': new_seed_sha,
        'terminal_seed_bytes': seed_path.stat().st_size,
        'terminal_codebook_count': 196,
        'shared_exact_count': 116,
        'canonical_dropped_count': 70,
        'physical_base_live_only_added_count': 12,
        'physical_base_live_only_bindings': bindings,
        'strict_subset_failure_status': str(status_path),
        'strict_subset_failure_status_sha256': status_sha,
        'terminal_keyspace_exact': True,
        'terminal_values_exact': True,
        'terminal_seed_extension_required': False,
        'old_to_new_alias_mapping_used': False,
        'training_updates_run': 0,
        'completed_unix': time.time(),
    })
    atomic_json(rebuild_path, rebuild)
    rebuild_sha = sha(rebuild_path)

    binding_path = ROOT / 'receipts/P959_TERMINAL_REPAIR_BINDING.json'
    binding = json.loads(binding_path.read_text())
    binding.update({
        'terminal_update_000_seed_sha256': new_seed_sha,
        'terminal_update_000_seed_rebuild_sha256': rebuild_sha,
        'terminal_update_000_actual_live_codebooks': 196,
        'terminal_update_000_extension_gate_required': False,
        'terminal_update_000_old_to_new_alias_mapping_used': False,
        'status': 'PASS_READY_TO_LAUNCH_UPDATE_001_FROM_COMPLETE_REBUILT_TERMINAL_UPDATE_000',
        'previous_binding_sha256': sha(binding_path),
        'completed_rebind_unix': time.time(),
    })
    atomic_json(binding_path, binding)
    binding_sha = sha(binding_path)

    ready_path = ROOT / 'receipts/P959_TERMINAL_LAUNCH_READY.json'
    ready = json.loads(ready_path.read_text())
    ready.update({
        'canonical_seed_sha256': new_seed_sha,
        'terminal_update_000_seed_rebuild_sha256': rebuild_sha,
        'binding_sha256': binding_sha,
        'terminal_update_000_actual_live_codebooks': 196,
        'terminal_update_000_extension_gate_required': False,
        'launcher_sha256': LAUNCHER_SHA,
        'status': 'PASS_READY_TO_LAUNCH_FROM_COMPLETE_REBUILT_TERMINAL_UPDATE_000',
        'previous_ready_sha256': sha(ready_path),
        'completed_rebind_unix': time.time(),
    })
    atomic_json(ready_path, ready)
    ready_sha = sha(ready_path)

    activation_path = ROOT / 'receipts/P959_TERMINAL_UPDATE_000_ACTIVATED.json'
    activation = json.loads(activation_path.read_text())
    activation.update({
        'terminal_seed_sha256': new_seed_sha,
        'seed_rebuild_sha256': rebuild_sha,
        'binding_sha256': binding_sha,
        'launch_ready_sha256': ready_sha,
        'actual_live_codebooks': 196,
        'extension_gate_required': False,
        'status': 'PASS_COMPLETE_REBUILT_TERMINAL_UPDATE_000_ACTIVATED',
        'previous_activation_sha256': sha(activation_path),
        'completed_activation_unix': time.time(),
    })
    atomic_json(activation_path, activation)
    activation_sha = sha(activation_path)

    receipt = {
        'schema': 'p959-terminal-update-000-actual-live-completion-v1',
        'status': 'PASS_COMPLETE_ACTUAL_LIVE_SEED_NO_ALIAS_NO_EXTENSION',
        'task_id': TASK,
        'host': 'spark-3',
        'claim_sha256': CLAIM_SHA,
        'p943_terminal_sha256': TERMINAL_SHA,
        'active_overlay_sha256': OVERLAY_SHA,
        'preimage_seed': str(preserved),
        'preimage_seed_sha256': OLD_SEED_SHA,
        'complete_seed': str(seed_path),
        'complete_seed_sha256': new_seed_sha,
        'codebooks_before': 184,
        'physical_live_only_added': 12,
        'codebooks_after': 196,
        'bindings': bindings,
        'old_to_new_alias_mapping_used': False,
        'speculative_seed_used': False,
        'training_updates_run': 0,
        'extension_gate_required': False,
        'rebuild_receipt': str(rebuild_path),
        'rebuild_receipt_sha256': rebuild_sha,
        'binding_sha256': binding_sha,
        'ready_sha256': ready_sha,
        'activation_sha256': activation_sha,
        'launcher_sha256': LAUNCHER_SHA,
        'trainer_sha256': TRAINER_SHA,
        'gpu_apps': [],
        'scoped_processes': [],
        'root_free_bytes': root_free,
        'sealed_unix': time.time(),
    }
    receipt_path = ROOT / 'receipts/P959_TERMINAL_UPDATE_000_ACTUAL_LIVE_COMPLETION.json'
    atomic_json(receipt_path, receipt)
    print(json.dumps({'receipt': str(receipt_path), 'receipt_sha256': sha(receipt_path), **receipt}, sort_keys=True))


if __name__ == '__main__':
    main()
