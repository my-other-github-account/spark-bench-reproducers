#!/usr/bin/env python3
"""VQ3-uniform plane reader with optional W3v2 fallback for partial coverage.

VQ3 planes use the vqa-compatible keys with int16 storage codes and d=4;
k is read from each layer's metadata (supported: 4096 or 8192):
  vq3u_layer_NNN.pt {codes13, sc13, cb13, codes2, sc2, cb2, meta}
Missing layers may delegate to the sealed shipped-plane reader named by
VQ3U_FALLBACK_W3; this is used only for the explicitly labelled partial row.
"""
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

import torch

MISSION_ROOT = os.path.expanduser(os.environ.get('MISSION_ROOT', '~/missions'))
TEACH = os.path.expanduser(os.environ.get(
    'DS4_TEACHER_ROOT', os.path.join(MISSION_ROOT, 'DS4_TEACHER')))
if TEACH not in sys.path:
    sys.path.insert(0, TEACH)
import t8192_ds4_build_v3 as v3  # noqa: E402

_SealedPlaneSource = v3.PlaneSource
_DIMS = (256, 4096, 4096, 4096, 2048)


def _receipt_md5(row):
    value = row.get('canonical_md5') or row.get('md5')
    if not value:
        raise ValueError('plane receipt row is missing canonical_md5/md5')
    return value


def _receipt_source_label(row, is_override):
    if is_override:
        return 'local_override'
    return row.get('canonical_source_host', 'canonical_remote')


class Vq3UniformBandSource:
    def __init__(self, planes_dir):
        self.dir = os.path.expanduser(planes_dir)
        self.stream_cache = os.environ.get('VQ3U_STREAM_CACHE', '').strip()
        self.stream_s8 = os.environ.get('VQ3U_STREAM_S8', '').rstrip('/')
        self.stream_s6 = os.environ.get('VQ3U_STREAM_S6', '').rstrip('/')
        self.stream_receipt = os.environ.get('VQ3U_STREAM_RECEIPT', '').strip()
        self.stream_ledger = os.environ.get('VQ3U_STREAM_LEDGER', '').strip()
        self.override_dir = os.environ.get('VQ3U_OVERRIDE_DIR', '').strip()
        self.override_receipt = os.environ.get('VQ3U_OVERRIDE_RECEIPT', '').strip()
        self.overrides = {}
        self.expected = {}
        self._cached_path = None
        if self.stream_cache:
            assert self.stream_s8 and self.stream_s6, 'both QSFP stream sources are required'
            assert self.stream_receipt and os.path.isfile(self.stream_receipt), self.stream_receipt
            os.makedirs(self.stream_cache, exist_ok=True)
            receipt = json.load(open(self.stream_receipt))
            markers = receipt['canonical_planes']['s1_marker_binding']['markers']
            self.expected = {int(x['layer']): x for x in markers}
            assert set(self.expected) == set(range(43)), sorted(self.expected)
            if self.override_dir:
                assert self.override_receipt and os.path.isfile(self.override_receipt), self.override_receipt
                override_rows = json.load(open(self.override_receipt))['layers']
                self.overrides = {int(x['layer']): x for x in override_rows}
                assert self.overrides and set(self.overrides) <= set(range(43))
            if not self.stream_ledger:
                self.stream_ledger = os.path.join(self.stream_cache, 'STREAMED_PLANES_VERIFICATION.jsonl')
        else:
            assert os.path.isdir(self.dir), self.dir
        fallback = os.environ.get('VQ3U_FALLBACK_W3', '').strip()
        self.fallback_dir = os.path.expanduser(fallback) if fallback else None
        self.fallback = _SealedPlaneSource(self.fallback_dir) if self.fallback_dir else None
        self._cache = {}

    def _path(self, layer):
        return os.path.join(self.dir, f'vq3u_layer_{layer:03d}.pt')

    @staticmethod
    def _md5(path):
        h = hashlib.md5()
        with open(path, 'rb') as f:
            for block in iter(lambda: f.read(8 << 20), b''):
                h.update(block)
        return h.hexdigest()

    def _record_stream(self, **row):
        row['ts'] = time.time()
        with open(self.stream_ledger, 'a') as f:
            f.write(json.dumps(row, sort_keys=True) + '\n')
            f.flush()
            os.fsync(f.fileno())

    def _stage_streamed_plane(self, layer):
        expected = self.overrides.get(layer, self.expected[layer])
        dst = os.path.join(self.stream_cache, f'vq3u_layer_{layer:03d}.pt')
        is_override = layer in self.overrides
        expected_md5 = _receipt_md5(expected)
        source_label = _receipt_source_label(expected, is_override)
        source_root = self.override_dir if is_override else (
            self.stream_s8 if layer <= 21 else self.stream_s6)
        source = (os.path.join(source_root, f'vq3u_layer_{layer:03d}.pt')
                  if is_override else f'{source_root}/vq3u_layer_{layer:03d}.pt')

        # Drop the previous mmap before unlinking its backing file. The builder
        # has already materialized and dematerialized that layer by this point.
        self._cache = {}
        gc.collect()
        if self._cached_path and self._cached_path != dst:
            try:
                os.remove(self._cached_path)
            except FileNotFoundError:
                pass
        self._cached_path = None

        def checked_md5(path):
            if not os.path.isfile(path) or os.path.getsize(path) != int(expected['bytes']):
                return None
            got = self._md5(path)
            return got if got == expected_md5 else None

        got_md5 = checked_md5(dst)
        if got_md5 is None:
            try:
                os.remove(dst)
            except FileNotFoundError:
                pass
            free = shutil.disk_usage(self.stream_cache).free
            if free < 12 << 30:
                raise RuntimeError(f'stream cache disk guard: only {free >> 30} GiB free')
            tmp = dst + '.partial'
            for attempt in range(1, 4):
                try:
                    os.remove(tmp)
                except FileNotFoundError:
                    pass
                if is_override:
                    try:
                        shutil.copyfile(source, tmp)
                        result = subprocess.CompletedProcess([], 0, '', '')
                    except Exception as exc:
                        result = subprocess.CompletedProcess([], 1, '', str(exc))
                else:
                    cmd = [
                        'scp', '-q', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                        '-o', 'ServerAliveInterval=15', source, tmp,
                    ]
                    result = subprocess.run(cmd, text=True, capture_output=True)
                tmp_md5 = checked_md5(tmp) if result.returncode == 0 else None
                if tmp_md5 is not None:
                    os.replace(tmp, dst)
                    got_md5 = tmp_md5
                    break
                print(
                    f'[Vq3UniformBandSource] stream L{layer:03d} attempt {attempt} '
                    f'failed rc={result.returncode}: {result.stderr[-200:]}',
                    flush=True,
                )
                time.sleep(5)
            else:
                raise RuntimeError(f'failed to fetch/verify canonical plane L{layer:03d} from {source}')

        assert got_md5 == expected_md5
        self._record_stream(
            layer=layer,
            source=source,
            path=dst,
            bytes=os.path.getsize(dst),
            md5=got_md5,
            expected_md5=expected_md5,
            source_kind='corrected_local_override' if is_override else 'canonical_remote',
            status='PASS',
        )
        self._cached_path = dst
        print(
            f'[Vq3UniformBandSource] streamed canonical L{layer:03d} '
            f'from {source_label} md5={got_md5}',
            flush=True,
        )
        return dst

    def _load(self, layer):
        if layer not in self._cache:
            path = self._stage_streamed_plane(layer) if self.stream_cache else self._path(layer)
            assert os.path.isfile(path), path
            self._cache = {
                layer: torch.load(path, map_location='cpu', mmap=True, weights_only=True)
            }
        return self._cache[layer]

    def layer(self, layer):
        path = self._path(layer)
        if not self.stream_cache and not os.path.isfile(path):
            assert self.fallback is not None, f'missing VQ3 plane L{layer:03d}: {path}'
            print(f'[Vq3UniformBandSource] L{layer:03d} -> W3v2 fallback', flush=True)
            return self.fallback.layer(layer)

        data = self._load(layer)
        meta = data.get('meta', {})
        k = int(meta.get('k', int(data['cb13'].shape[0])))
        d = int(meta.get('d', int(data['cb13'].shape[1])))
        assert k in (4096, 8192), meta
        assert d == 4, meta
        assert int(data['cb13'].shape[0]) == k and int(data['cb2'].shape[0]) == k
        assert data['codes13'].dtype == torch.int16
        assert data['codes2'].dtype == torch.int16
        print(f'[Vq3UniformBandSource] L{layer:03d} -> VQ3U k{k}/d{d}', flush=True)

        def expert(expert_id, which):
            key = '13' if which == '13' else '2'
            codes = data[f'codes{key}'][expert_id].to(v3.DEV)
            scales = data[f'sc{key}'][expert_id].to(v3.DEV)
            codebook = data[f'cb{key}'].to(v3.DEV).float()
            scale_columns = torch.exp2(scales.float() - 127.0).repeat_interleave(32, dim=1)
            weights = codebook[codes.long()].reshape(codes.shape[0], -1)
            return weights * scale_columns

        return expert, _DIMS


def self_test(planes_dir, fallback_dir=None):
    if fallback_dir:
        os.environ['VQ3U_FALLBACK_W3'] = fallback_dir
    src = Vq3UniformBandSource(planes_dir)
    covered = sorted(
        int(name[11:14]) for name in os.listdir(src.dir)
        if name.startswith('vq3u_layer_') and name.endswith('.pt')
    )
    assert covered
    for layer in (covered[0], covered[-1]):
        expert, dims = src.layer(layer)
        assert dims == _DIMS
        for expert_id, which, shape in ((0, '13', (4096, 4096)), (255, '2', (4096, 2048))):
            got = expert(expert_id, which)
            assert tuple(got.shape) == shape
            assert torch.isfinite(got).all()
            print(f'self-test VQ3 L{layer:03d} e{expert_id} {which}: {shape} {got.dtype} PASS')
            del got
            torch.cuda.empty_cache()
    if src.fallback is not None:
        missing = next(layer for layer in range(43) if layer not in covered)
        expert, dims = src.layer(missing)
        got = expert(0, '13')
        assert dims == _DIMS and tuple(got.shape) == (4096, 4096)
        assert torch.isfinite(got).all()
        print(f'self-test fallback L{missing:03d} e0 13: {tuple(got.shape)} {got.dtype} PASS')
    print(f'VQ3U source self-test PASS coverage={len(covered)}/43 layers={covered}')


if __name__ == '__main__':
    self_test(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
