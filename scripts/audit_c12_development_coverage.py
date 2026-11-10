"""Locate missing truths in existing retrieval pools using only40 used dev labels."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import collections
import json
from datetime import datetime, timezone
from functools import lru_cache

import numpy as np
import pandas as pd
import pyarrow as pa
from rdkit import rdBase

from precision_cache import ROOT, sha, write, features

OUT = ROOT / 'outputs/local/c12-coverage-20261110'
CLOUD = ROOT / 'outputs/forks/c10-casmi2022-validation-v1'
POOLS = ROOT / 'outputs/forks/c10-candidate-pools-v1'
LOCAL = ROOT / 'outputs/local/c10-casmi2022-20261110'


def main():
    pa.set_cpu_count(2); assert rdBase.rdkitVersion == '2026.03.3'
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'coverage.json').exists(), 'Completed development diagnostic is sealed'
    selection = ROOT / 'outputs/local/c10-candidates-20261110/selection-freeze.json'
    assert not json.loads(selection.read_text())['new_candidate_selected']
    paths = [CLOUD / 'c10_p01.csv', CLOUD / 'c10_neural_p01.parquet', CLOUD / 'outsider_neural_features.parquet',
        POOLS / 'eng_lists.json', POOLS / 'pc_lists.json', POOLS / 'outsider_evidence.parquet']
    paths += [POOLS / name / (name.split('_')[0] + '_in.json') for name in ['ice_work', 'gl_work', 'n1_work']]
    write(OUT / 'coverage-protocol.json', dict(prepared_utc=datetime.now(timezone.utc).isoformat(),
        auditor_sha256=sha(__file__), feature_sha256=sha(ROOT / 'outputs/local/kit/features.py'),
        C10_selection_sha256=sha(selection), labels_sha256=sha(LOCAL / 'labels.csv'),
        files={str(path.relative_to(ROOT)).replace('\\', '/'): sha(path) for path in paths},
        scope='All40 already-used development molecules, absent truths retained; candidate coverage only, not a chosen recipe or top25 performance claim',
        reserved_prediction_metrics_read=False, automatic_submission=False))
    labels = pd.read_csv(LOCAL / 'labels.csv', dtype=str, nrows=40).set_index('molecule_id')
    assert len(labels) == 40 and labels.partition.eq('development').all()
    assert labels.index.tolist() == json.loads((LOCAL / 'partitions.json').read_text())['development']
    ids = set(labels.index)
    baseline = pd.read_csv(CLOUD / 'c10_p01.csv', dtype=str, keep_default_na=False).set_index('molecule_id').loc[labels.index]
    known = {}
    for name in ['c10_neural_p01.parquet', 'outsider_neural_features.parquet']:
        frame = pd.read_parquet(CLOUD / name, columns=['molecule_id', 'smiles', 'metric_key'])
        frame = frame[frame.molecule_id.isin(ids)]
        for smiles, key in zip(frame.smiles, frame.metric_key):
            assert smiles not in known or known[smiles] == key
            known[str(smiles)] = str(key)
    for smiles in sorted(known)[:32]: assert features.structure(smiles)[0] == known[smiles]
    invalid = set()
    @lru_cache(maxsize=20000)
    def identity(smiles):
        if smiles in known: return known[smiles]
        try: return features.structure(smiles)[0]
        except (ValueError, TypeError): invalid.add(smiles); return None
    pools = {name: {mid: [] for mid in labels.index} for name in ['P01_final', 'engine_top40', 'pubchem_top25', 'outsider_pool', 'ice_input_pool', 'gl_input_pool', 'n1_input_pool']}
    for mid, row in baseline.iterrows(): pools['P01_final'][mid] = row.smiles.split(';')
    eng = json.loads((POOLS / 'eng_lists.json').read_text()); pc = json.loads((POOLS / 'pc_lists.json').read_text())
    supplied_key_mismatches = 0
    for mid in labels.index:
        for name, source, smiles_field, key_field in [('engine_top40', eng, 'smiles', 'keys'), ('pubchem_top25', pc, 'pc', 'pc_keys')]:
            values = source.get(mid, {}); smiles = values.get(smiles_field, []); keys = values.get(key_field, [])
            assert len(smiles) == len(keys)
            pools[name][mid] = smiles
            supplied_key_mismatches += sum(identity(s) != k for s, k in zip(smiles, keys))
    extra = pd.read_parquet(POOLS / 'outsider_evidence.parquet')
    extra = extra[extra.molecule_id.isin(ids)]
    for mid, group in extra.groupby('molecule_id', sort=False): pools['outsider_pool'][mid] = group.smiles.tolist()
    for stage in ['ice', 'gl', 'n1']:
        rows = json.loads((POOLS / (stage + '_work') / (stage + '_in.json')).read_text())
        for row in rows:
            if str(row['mid']) in ids:
                assert isinstance(row['cands'], list) and all(isinstance(s, str) for s in row['cands'])
                pools[stage + '_input_pool'][str(row['mid'])] = row['cands']
    records = []; summaries = {}
    for mid in labels.index:
        truth = labels.loc[mid, 'key']; keysets = {}
        for name, source in pools.items(): keysets[name] = {key for s in source[mid] if (key := identity(s)) is not None}
        keysets['final_plus_outsiders'] = keysets['P01_final'] | keysets['outsider_pool']
        keysets['all_existing_pools'] = set().union(*keysets.values())
        row = dict(molecule_id=mid)
        for name, keys in keysets.items():
            row[name + '_truth_present'] = truth in keys; row[name + '_identities'] = len(keys)
        records.append(row)
    predictions = pd.DataFrame(records)
    for name in keysets:
        present = predictions[name + '_truth_present'].to_numpy()
        base = predictions.P01_final_truth_present.to_numpy()
        summaries[name] = dict(queries=40, truth_present=int(present.sum()), truth_absent=int((~present).sum()),
            absent_final_truths_recoverable=int((present & ~base).sum()),
            median_unique_candidates=float(predictions[name + '_identities'].median()),
            maximum_unique_candidates=int(predictions[name + '_identities'].max()))
    assert summaries['P01_final']['truth_present'] == 20
    predictions.to_csv(OUT / 'development-coverage.csv', index=False)
    report = dict(completed_utc=datetime.now(timezone.utc).isoformat(), development_queries=40,
        C10_development_already_used=True, coverage=summaries, invalid_structure_strings=len(invalid),
        supplied_engine_pubchem_key_mismatches=supplied_key_mismatches,
        coverage_protocol_sha256=sha(OUT / 'coverage-protocol.json'), row_audit_sha256=sha(OUT / 'development-coverage.csv'),
        reserved_labels_read=False, reserved_prediction_metrics_computed=False, model_scores_computed=False,
        recipe_selected=False, submission_qualified=False, automatic_submission=False,
        limitation='Oracle candidate presence in already-used development pools is not achievable top25 MRR or hidden improvement. Frozen C10 confirmation remains unused; any new recipe must freeze before reserved evaluation.')
    write(OUT / 'coverage.json', report)
    print('DONE C12 DEVELOPMENT COVERAGE', json.dumps(report), flush=True)


if __name__ == '__main__': main()
