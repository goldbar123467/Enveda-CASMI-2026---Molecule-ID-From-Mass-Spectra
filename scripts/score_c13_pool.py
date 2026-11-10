"""Score the deeper pool with unchanged P01 weights; no query labels or metrics."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2'); os.environ.setdefault('MKL_NUM_THREADS', '2')
import argparse
import gc
import importlib.util
import json
import shutil
from datetime import datetime, timezone, timedelta

import torch
import numpy as np
import pandas as pd
import pyarrow as pa

from precision_cache import ROOT, sha, write
from prepare_c13_pool import OUT, CONFIG, cloud
from reference_model_audit import verify
from training_monitor import system_memory


def stamp(): return datetime.now(timezone.utc).isoformat()


def main(scope):
    import msvcrt
    out = OUT / scope
    CLOUD = cloud(scope)
    lock = (out / 'scoring.lock').open('a+b'); lock.write(b'0'); lock.flush(); lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    config = json.loads(CONFIG.read_text()); contract = json.loads((OUT / 'source-contract.json').read_text())
    assert sha(CONFIG) == contract['config_sha256']
    for name, pin in contract['source_sha256'].items(): assert sha(ROOT / 'scripts' / name) == pin, name
    prep = json.loads((out / 'preparation.json').read_text())
    assert prep['source_contract_sha256'] == sha(OUT / 'source-contract.json')
    for name, pin in prep['files'].items(): assert sha(out / name) == pin, name
    assert not (out / 'scoring-receipt.json').exists(), 'Complete inference is sealed'
    assert not (ROOT / '.local/compute-paused.json').exists()
    torch.set_num_threads(2); torch.set_num_interop_threads(1); pa.set_cpu_count(2)
    assert torch.cuda.is_available() and '2070' in torch.cuda.get_device_name()
    torch.cuda.set_per_process_memory_fraction(config['gpu_memory_fraction'])
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    if not (out / 'scoring-launch.json').exists(): write(out / 'scoring-launch.json', dict(started_utc=stamp()))
    deadline = datetime.fromisoformat(json.loads((out / 'scoring-launch.json').read_text())['started_utc']) + timedelta(hours=config['maximum_scoring_hours'])
    def status(stage, **values):
        free = system_memory()['free_gib']; assert free >= config['minimum_free_ram_gib']
        assert datetime.now(timezone.utc) < deadline and not (ROOT / '.local/compute-paused.json').exists()
        row = dict(updated_utc=stamp(), pid=os.getpid(), stage=stage, free_ram_gib=free,
            gpu_peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30, **values)
        write(out / 'status.json', row); print(json.dumps(row), flush=True)
    loader_path = ROOT / 'outputs/research/c08-pipeline/modules/neural_scores.py'
    assert sha(loader_path) == contract['loader_sha256']
    spec = importlib.util.spec_from_file_location('_c13_existing_p01_loader', loader_path)
    loader = importlib.util.module_from_spec(spec); spec.loader.exec_module(loader)
    audit_path = ROOT / 'outputs/forks/p16-external-reference-c09-v2/p16_execution_receipt.json'
    assert sha(audit_path) == contract['model_audit_sha256']
    pins = json.loads(audit_path.read_text())['baseline_model_source_audit']
    public = ROOT / 'outputs/local/kit'; second = OUT / 'v80-bundle'; second.mkdir(exist_ok=True)
    for name in ['features.py', 'model.py', 'config.json']: shutil.copy2(public / name, second / name)
    checkpoint = ROOT / 'outputs/research/v80-output/casmi_spectrum_model_ours_v80/final_model.pt'
    if not (second / 'final_model.pt').exists(): shutil.copy2(checkpoint, second / 'final_model.pt')
    model_audit = verify(public, second, pins['public_files'], pins['second_checkpoint_sha256'])
    frame = pd.read_parquet(out / 'pool.parquet'); queries = pd.read_parquet(out / 'queries.parquet')
    lookup = {str(mid): group for mid, group in queries.groupby('molecule_id', sort=False)}
    fingerprints = np.load(out / 'fingerprints.npy', mmap_mode='r')
    assert fingerprints.shape[1] == 2048
    results = {}; parity = {}
    torch.cuda.reset_peak_memory_stats()
    for label, bundle, reference_name in [('public', public, 'neural_public.parquet'), ('v80', second, 'neural_ours.parquet')]:
        status('loading_' + label)
        model, features, cfg = loader.load_model(bundle, 'final_model.pt', 'cuda')
        assert cfg['fp_bits'] == 2048
        parts = []
        with torch.inference_mode():
            for number, (mid, group) in enumerate(frame.groupby('molecule_id', sort=False)):
                group = group.sort_values('prior_rank', kind='stable').copy(); records = []
                for row in lookup[str(mid)].itertuples(index=False):
                    peaks, length = features.peaks(row.ms2_mzs, row.ms2_normalized_intensities, row.precursor_mz, cfg['max_peaks'])
                    if length < cfg['min_peaks']: continue
                    metadata, adduct = features.metadata(row.precursor_mz, row.adduct, row.ionization_mode, row.collision_energy_ev)
                    records.append((peaks, length, metadata, adduct))
                assert records, 'No supported spectra for ' + str(mid)
                fp = torch.tensor(np.asarray(fingerprints[group.fp_index.to_numpy()]), device='cuda', dtype=torch.float32)
                molecules = model.encode_molecules(fp); cosines = []; fingerprint_scores = []
                for start in range(0, len(records), 64):
                    batch = records[start:start+64]
                    peaks = torch.tensor(np.stack([v[0] for v in batch]), device='cuda')
                    lengths = torch.tensor([v[1] for v in batch], device='cuda', dtype=torch.long)
                    metadata = torch.tensor(np.stack([v[2] for v in batch]), device='cuda')
                    adducts = torch.tensor([v[3] for v in batch], device='cuda', dtype=torch.long)
                    z, predicted_fp = model.encode_spectra(peaks, lengths, metadata, adducts)
                    cosines.append((z @ molecules.T).cpu().numpy())
                    fingerprint_scores.append((predicted_fp @ fp.T / cfg['fp_bits']).cpu().numpy())
                cosine = np.concatenate(cosines); fps = np.concatenate(fingerprint_scores)
                combined = cosine / cfg['temperature'] + cfg['fp_score_weight'] * fps
                assert np.isfinite(combined).all()
                group['neural_score'] = combined.mean(0); group['neural_ok'] = True
                group['neural_cos'] = cosine.mean(0); group['neural_fp'] = fps.mean(0)
                parts.append(group)
                if (number+1) % 10 == 0: status('scoring_' + label, completed_queries=number+1, total_queries=prep['queries'])
        result = pd.concat(parts, ignore_index=True)
        reference = pd.read_parquet(CLOUD / reference_name)
        reference = reference[reference.molecule_id.isin(set(frame.molecule_id))].set_index(['molecule_id', 'metric_key'])
        assert reference.index.is_unique and reference.neural_ok.astype(bool).all()
        local = result.set_index(['molecule_id', 'metric_key']).loc[reference.index]
        difference = float((local.neural_score - reference.neural_score).abs().max())
        parity[label] = dict(rows=len(reference), max_raw_score_abs_difference=difference,
            tolerance=config['local_raw_score_parity_tolerance'])
        result.to_parquet(out / (label + '.parquet'), index=False)
        assert difference <= config['local_raw_score_parity_tolerance'], 'Local/cloud raw score mismatch: ' + label
        results[label] = result
        del model; gc.collect(); torch.cuda.empty_cache()
    result = results['public'].copy()
    other = results['v80'].set_index(['molecule_id', 'metric_key']).loc[result.set_index(['molecule_id', 'metric_key']).index]
    result['public_score'] = results['public']['neural_score'].to_numpy()
    result['v80_score'] = other.neural_score.to_numpy()
    result['neural_score'] = 0.
    for _, positions in result.groupby('molecule_id', sort=False).groups.items():
        indices = np.asarray(list(positions)); a = results['public'].loc[indices, 'neural_score'].to_numpy(); b = other.iloc[indices].neural_score.to_numpy()
        result.loc[indices, 'neural_score'] = .5*(a-a.mean())/max(float(a.std()),1e-6) + .5*(b-b.mean())/max(float(b.std()),1e-6)
    result.to_parquet(out / 'scores.parquet', index=False)
    write(out / 'scoring-receipt.json', dict(completed_utc=stamp(), scope=scope, models=model_audit,
        source_contract_sha256=sha(OUT / 'source-contract.json'), preparation_sha256=sha(out / 'preparation.json'),
        queries=int(result.molecule_id.nunique()), candidates=len(result), input_errors=0, local_cloud_parity=parity,
        scores_sha256=sha(out / 'scores.parquet'), gpu_peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
        query_labels_read=False, prediction_metrics_computed=False, automatic_submission=False))
    status('complete', completed_models=2); print('DONE C13 UNLABELLED SCORING', scope, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--scope', choices=['development', 'confirmation'], default='development')
    args = parser.parse_args()
    try: main(args.scope)
    except Exception as exc:
        write(OUT / args.scope / 'attention.json', dict(checked_utc=stamp(), error=repr(exc), automatic_submission=False))
        raise
