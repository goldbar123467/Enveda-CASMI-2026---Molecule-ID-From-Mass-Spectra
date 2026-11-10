"""Build a label-blind deeper pool; development and all160 outputs are isolated."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2'); os.environ.setdefault('MKL_NUM_THREADS', '2')
import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow as pa
from rdkit import rdBase

from precision_cache import ROOT, sha, write, features
from external_reference_overlay import apply_overlay
from training_monitor import system_memory

OUT = ROOT / 'outputs/local/c12-deep-candidates-20261110'
CONFIG = ROOT / 'configs/c12-deep-candidates-20261110.json'
CLOUD = ROOT / 'outputs/forks/c10-casmi2022-validation-v1'
POOLS = ROOT / 'outputs/forks/c10-candidate-pools-v1'


def main(scope):
    pa.set_cpu_count(2); assert rdBase.rdkitVersion == '2026.03.3'
    cfg = json.loads(CONFIG.read_text()); assert system_memory()['free_gib'] >= cfg['minimum_free_ram_gib']
    assert not (ROOT / '.local/compute-paused.json').exists()
    out = OUT / scope; out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'preparation.json').exists(), 'Prepared scope is sealed'
    sources = ['prepare_c12_pool.py', 'score_c12_pool.py', 'prefix_expansion.py']
    contract = dict(config_sha256=sha(CONFIG), source_sha256={n: sha(ROOT / 'scripts' / n) for n in sources},
        feature_sha256=sha(ROOT / 'outputs/local/kit/features.py'),
        loader_sha256=sha(ROOT / 'outputs/research/c08-pipeline/modules/neural_scores.py'),
        model_audit_sha256=sha(ROOT / 'outputs/forks/p16-external-reference-c09-v2/p16_execution_receipt.json'),
        P16_policy_sha256=sha(ROOT / 'configs/c09-external-reference-20261109.json'))
    if (OUT / 'source-contract.json').exists(): assert json.loads((OUT / 'source-contract.json').read_text()) == contract
    else: write(OUT / 'source-contract.json', contract)
    baseline = pd.read_csv(CLOUD / 'c10_p01.csv', dtype=str, keep_default_na=False)
    partitions = json.loads((ROOT / 'outputs/local/c10-casmi2022-20261110/partitions.json').read_text())
    ids = partitions['development'] if scope == 'development' else baseline.molecule_id.tolist()
    baseline = baseline.set_index('molecule_id').loc[ids].reset_index(); wanted = set(ids)
    original = pd.read_parquet(CLOUD / 'c10_neural_p01.parquet')
    original = original[original.molecule_id.isin(wanted)]
    assert original.groupby('molecule_id').library_max.nunique().max() == 1
    confidence = original.groupby('molecule_id').library_max.first().to_dict()
    cache = {}; invalid = set()
    def structure(smiles):
        if smiles not in cache:
            try: cache[smiles] = features.structure(smiles, 2048)
            except (ValueError, TypeError): invalid.add(smiles); return None
        return cache[smiles]
    def identity(smiles):
        value = structure(smiles)
        if value is None: raise ValueError('Invalid candidate')
        return value[0]
    matches = pd.read_parquet(ROOT / 'outputs/local/c10-candidates-20261110/matches.parquet')
    matches = matches[matches.recipe.eq('P16') & matches.view.eq('all_reference') & matches.molecule_id.isin(wanted)]
    parent, _ = apply_overlay(baseline, matches, confidence, json.loads((ROOT / 'configs/c09-external-reference-20261109.json').read_text()), identity)
    if scope == 'development':
        expected = pd.read_csv(ROOT / 'outputs/local/c10-candidates-20261110/development-P16-all_reference.csv', dtype=str, keep_default_na=False)
        assert parent.equals(expected)
    eng = json.loads((POOLS / 'eng_lists.json').read_text()); pc = json.loads((POOLS / 'pc_lists.json').read_text())
    inputs = {stage: {str(row['mid']): row['cands'] for row in json.loads((POOLS / (stage+'_work') / (stage+'_in.json')).read_text())} for stage in ['ice', 'gl', 'n1']}
    outsider = pd.read_parquet(POOLS / 'outsider_evidence.parquet')
    records = []
    for number, row in enumerate(parent.itertuples(index=False)):
        mid = row.molecule_id; before = row.smiles.split(';'); taken = set(); pool = []
        def add(smiles, source, rank, is_parent=False):
            value = structure(smiles)
            if value is None:
                assert not is_parent, 'Invalid parent structure'; return
            key = value[0]
            if key in taken: return
            taken.add(key); pool.append(dict(molecule_id=mid, smiles=smiles, metric_key=key,
                prior_rank=len(pool)+1, is_parent=is_parent, library_max=float(confidence[mid]),
                processed=True, source=source, source_rank=rank))
        for rank, smiles in enumerate(before, 1): add(smiles, 'parent_P16', rank, True)
        assert len(pool) == len(before) and len(before) <= 25
        pre = original[original.molecule_id.eq(mid)].sort_values('baseline_rank', kind='stable')
        for rank, smiles in enumerate(pre.smiles, 1): add(smiles, 'original_neural', rank)
        extra = outsider[outsider.molecule_id.eq(mid)].sort_values(['source_rank', 'source'], kind='stable').smiles.tolist()
        lists = [eng.get(mid, {}).get('smiles', []), pc.get(mid, {}).get('pc', []), inputs['ice'].get(mid, []), inputs['gl'].get(mid, []), inputs['n1'].get(mid, []), extra]
        for rank in range(max(map(len, lists), default=0)):
            for source, values in zip(cfg['candidate_sources_in_round_robin_order'], lists):
                if len(pool) >= cfg['maximum_total_candidates']: break
                if rank < len(values): add(str(values[rank]), source, rank+1)
            if len(pool) >= cfg['maximum_total_candidates']: break
        records.extend(pool)
        if (number+1) % 5 == 0:
            assert system_memory()['free_gib'] >= cfg['minimum_free_ram_gib']
            print('C12 POOL PREPARED', number+1, '/', len(parent), flush=True)
    evidence = pd.DataFrame(records); smiles = evidence.smiles.drop_duplicates().tolist()
    fp_index = {s: i for i, s in enumerate(smiles)}; evidence['fp_index'] = evidence.smiles.map(fp_index)
    np.save(out / 'fingerprints.npy', np.stack([cache[s][-1] for s in smiles]).astype(np.uint8))
    pd.DataFrame(dict(smiles=smiles, metric_key=[cache[s][0] for s in smiles])).to_parquet(out / 'fingerprint-smiles.parquet', index=False)
    evidence.to_parquet(out / 'pool.parquet', index=False); parent.to_csv(out / 'parent.csv', index=False)
    queries = pd.read_parquet(CLOUD / 'c10_queries.parquet'); queries[queries.molecule_id.isin(wanted)].to_parquet(out / 'queries.parquet', index=False)
    files = ['pool.parquet', 'parent.csv', 'queries.parquet', 'fingerprints.npy', 'fingerprint-smiles.parquet']
    write(out / 'preparation.json', dict(completed_utc=datetime.now(timezone.utc).isoformat(), scope=scope,
        queries=len(ids), candidates=len(evidence), unique_smiles=len(smiles), invalid_extra_strings=len(invalid),
        source_contract_sha256=sha(OUT / 'source-contract.json'), files={n: sha(out/n) for n in files},
        query_labels_read=False, prediction_metrics_computed=False, automatic_submission=False))
    print('DONE C12 POOL PREPARATION', scope, len(ids), len(evidence), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--scope', choices=['development', 'all'], default='development')
    main(parser.parse_args().scope)
