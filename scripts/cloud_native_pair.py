"""Run general/native paired trials on the two T4s; preprocessing stays off the workstation."""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def native_index(cache, native_file, kit, output):
    sys.path.insert(0, str(kit))
    from features import peaks, metadata
    mol = pd.read_parquet(cache / 'molecules.parquet')
    known = set(mol.key.astype(str))
    digests, raw_rows, prepared, unknown, invalid = set(), 0, 0, 0, 0
    source = pq.ParquetFile(native_file)
    cols = ['inchikey14', 'ms2_mzs', 'ms2_normalized_intensities', 'precursor_mz', 'adduct', 'ionization_mode', 'collision_energy_ev']
    for batch in source.iter_batches(batch_size=4096, columns=cols):
        for row in batch.to_pandas().itertuples(index=False):
            raw_rows += 1
            key = str(row.inchikey14)
            if key not in known:
                unknown += 1
                continue
            try:
                precursor = float(row.precursor_mz)
                if not np.isfinite(precursor) or precursor <= 0:
                    invalid += 1
                    continue
                x, length = peaks(row.ms2_mzs, row.ms2_normalized_intensities, precursor, 96)
                if length < 3:
                    invalid += 1
                    continue
                meta, adduct = metadata(precursor, row.adduct, row.ionization_mode, row.collision_energy_ev)
                digest = hashlib.sha256(key.encode() + x.astype('<f2').tobytes() + meta.astype('<f4').tobytes() + bytes([adduct])).digest()
                digests.add(digest)
                prepared += 1
            except (ValueError, TypeError):
                invalid += 1
        if raw_rows % (4096 * 25) == 0:
            print('NATIVE PREP', raw_rows, 'raw rows', len(digests), 'unique inputs', flush=True)
    spec = pd.read_parquet(cache / 'spectra.parquet')
    n = len(spec)
    x = np.memmap(cache / 'peaks.bin', mode='r', dtype='<f2', shape=(n, 96, 4))
    meta = np.memmap(cache / 'metadata.bin', mode='r', dtype='<f4', shape=(n, 4))
    keys, adducts = spec.key.to_numpy(), spec.adduct_id.to_numpy()
    matched, seen = [], set()
    for i in range(n):
        digest = hashlib.sha256(str(keys[i]).encode() + x[i].tobytes() + meta[i].tobytes() + bytes([int(adducts[i])])).digest()
        if digest in digests:
            matched.append(i)
            seen.add(digest)
        if (i + 1) % 500000 == 0:
            print('NATIVE CACHE MATCH', i + 1, 'of', n, 'matched', len(matched), flush=True)
    coverage = len(seen) / len(digests)
    audit = dict(raw_rows=raw_rows, prepared=prepared, unknown_raw_identity=unknown, invalid=invalid,
                 unique_native_inputs=len(digests), matched_unique=len(seen), cache_spectra=n,
                 matched_cache_rows=len(matched), coverage=coverage,
                 contract='SHA256(canonical key + f16 peak tensor + f32 metadata + adduct), identical to training cache')
    (output.parent / 'native_index_audit.json').write_text(json.dumps(audit, indent=2))
    assert coverage >= .995, audit
    np.save(output, np.asarray(matched, dtype=np.int64), allow_pickle=False)
    print('NATIVE INDEX READY', json.dumps(audit), flush=True)


def main():
    started = time.time()
    inputs, work = Path('/kaggle/input'), Path('/kaggle/working')
    cache = next(p.parent for p in inputs.rglob('peaks.bin') if p.parent.name == 'cache')
    kit = next(p.parent for p in inputs.rglob('train.py') if p.parent.name == 'casmi_spectrum_model_v1')
    raw = next(inputs.rglob('native_training_spectra.parquet'))
    target = np.load(next(inputs.rglob('np_unseen_candidate_keys.npz')), allow_pickle=False)
    mol = pd.read_parquet(cache / 'molecules.parquet')
    excluded = sorted(set(mol.loc[mol.split.eq('validation'), 'key'].astype(str)) | set(target.files))
    roots = {mode: work / 'paired' / mode for mode in ['general', 'native']}
    for folder in roots.values():
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'np_holdout_keys.txt').write_text('\n'.join(excluded) + '\n')
    print('UNION EXCLUSION', len(excluded), 'identities; SHA256', hashlib.sha256('\n'.join(excluded).encode()).hexdigest(), flush=True)
    processes, handles = {}, []

    def start(mode, gpu, index=None):
        env = dict(os.environ, CASMI_KIT=str(kit), CASMI_CACHE=str(cache), CASMI_OUT=str(roots[mode]),
                   CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
        log = (work / f'{mode}.log').open('w')
        handles.append(log)
        args = [sys.executable, '-u', str(Path(__file__).with_name('native_training.py')), '--mode', mode, '--seed', '21', '--epochs', '128']
        if index is not None:
            args += ['--native-index', str(index)]
        processes[mode] = subprocess.Popen(args, env=env, stdout=log, stderr=subprocess.STDOUT)
        print('TRAINING STARTED', mode, 'GPU', gpu, 'pid', processes[mode].pid, flush=True)

    start('general', 0)
    index = work / 'native_rows.npy'
    native_index(cache, raw, kit, index)
    start('native', 1, index)
    while any(process.poll() is None for process in processes.values()):
        for mode in processes:
            lines = (work / f'{mode}.log').read_text(errors='replace').splitlines()
            print(mode, lines[-1][:600] if lines else 'starting', flush=True)
        time.sleep(60)
    for handle in handles:
        handle.close()
    for mode, process in processes.items():
        assert process.returncode == 0, (mode, process.returncode, (work / f'{mode}.log').read_text()[-3000:])
        report = json.loads((roots[mode] / 'holdout_np_report.json').read_text())
        assert len(report) == 1
        result = next(iter(report.values()))
        assert result['completed_epochs'] == 128 and len(result['history']) == 128
        assert all(row['steps'] == 500 for row in result['history'])
    print('PAIRED TRAINING COMPLETE', round(time.time() - started), 's', flush=True)


if __name__ == '__main__':
    main()
