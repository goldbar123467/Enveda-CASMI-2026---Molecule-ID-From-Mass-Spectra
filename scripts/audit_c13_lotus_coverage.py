"""Label-blind LOTUS retrieval, followed by oracle coverage on used C10 data.

All160 C10 identities are already used after C12 confirmation. No fresh C13
labels or predictions are read. Oracle coverage is not an achievable MRR gain.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import collections
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow as pa
from rdkit import rdBase

from precision_cache import ROOT, sha, write, features
from build_external_cache import OFFSETS, normalize_adduct
from training_monitor import system_memory

OUT = ROOT / 'outputs/local/c13-lotus-development-20261110'
SOURCE = ROOT / 'outputs/research/c12-lotus-20261110'
OLD = ROOT / 'outputs/local/c12-deep-candidates-20261110'
PARAMETERS = dict(precursor_ppm=20.,precursor_absolute_da=.003,
    maximum_lotus_candidates_per_query=128,known_adducts_only=True,
    candidate_order='Absolute predicted precursor error, then mirror row; official metric identity dedup',
    unsupported='No candidates; never default an unknown adduct to protonation')


def main():
    pa.set_cpu_count(2); assert rdBase.rdkitVersion == '2026.03.3' and system_memory()['free_gib'] >= 4
    assert not (OUT / 'coverage.json').exists() and not (OUT / 'protocol.json').exists()
    OUT.mkdir(exist_ok=True)
    audit_path = ROOT / 'outputs/local/c13-lotus-source-20261110/audit.json'
    audit = json.loads(audit_path.read_text()); assert audit['traced_mirror_rows'] == 149983 and audit['unresolved_mirror_rows'] == 0
    consumed = json.loads((OLD / 'confirmation.json').read_text())
    assert consumed['queries'] == 120 and not consumed['statistical_gate']['passed']
    cloud = ROOT / 'outputs/forks/c10-casmi2022-validation-v1'
    oldprep = json.loads((OLD / 'all/preparation.json').read_text())
    assert sha(OLD / 'all/pool.parquet') == oldprep['files']['pool.parquet']
    write(OUT / 'protocol.json',dict(prepared_utc=datetime.now(timezone.utc).isoformat(),
        parameters=PARAMETERS,auditor_sha256=sha(__file__),source_audit_sha256=sha(audit_path),
        catalogue_sha256=sha(SOURCE / 'candidates_lotus.parquet'),queries_sha256=sha(cloud / 'c10_queries.parquet'),
        old_pool_sha256=sha(OLD / 'all/pool.parquet'),C12_confirmation_sha256=sha(OLD / 'confirmation.json'),
        C10_all160_already_used=True,C13_fresh_labels_read=False,model_scores_computed=False,
        method='Retrieve and seal all candidate/feature artifacts before opening used labels; no truth injection or coefficient search'))
    catalogue = pd.read_parquet(SOURCE / 'candidates_lotus.parquet'); masses = catalogue.monoisotopic_mass.to_numpy()
    assert np.isfinite(masses).all() and np.all(masses[1:] >= masses[:-1])
    queries = pd.read_parquet(cloud / 'c10_queries.parquet'); old = pd.read_parquet(OLD / 'all/pool.parquet')
    assert queries.molecule_id.is_unique and len(queries) == 160
    cache = {}; records = []; counts = collections.Counter()
    for number,row in enumerate(queries.itertuples(index=False)):
        adduct = normalize_adduct(row.adduct)
        mode = str(row.ionization_mode).lower()
        if adduct not in OFFSETS or (adduct.endswith('+') != (mode == 'positive')):
            counts['unsupported_adduct_or_mode'] += 1; continue
        charge, offset = OFFSETS[adduct]; neutral = row.precursor_mz*charge-offset
        width = (row.precursor_mz*PARAMETERS['precursor_ppm']*1e-6+PARAMETERS['precursor_absolute_da'])*charge
        positions = np.arange(np.searchsorted(masses,neutral-width,'left'),np.searchsorted(masses,neutral+width,'right'))
        positions = sorted(positions,key=lambda i:(abs((masses[i]+offset)/charge-row.precursor_mz),int(i)))
        taken = set(); rank = 0
        for index in positions:
            item = catalogue.iloc[index]; smiles = str(item.smiles)
            if smiles not in cache:
                try: cache[smiles] = features.structure(smiles,2048)
                except (ValueError,TypeError): cache[smiles] = None; counts['invalid_feature_structure'] += 1
            value = cache[smiles]
            if value is None or value[0] in taken: continue
            taken.add(value[0]); rank += 1
            records.append(dict(molecule_id=row.molecule_id,smiles=smiles,metric_key=value[0],
                mirror_row=int(index),source_rank=rank,within_fixed_cap=rank <= PARAMETERS['maximum_lotus_candidates_per_query'],
                provided_raw_key=str(item.inchikey14),neutral_mass=float(item.monoisotopic_mass),
                precursor_error_da=float(abs((masses[index]+offset)/charge-row.precursor_mz))))
        if (number+1) % 20 == 0: print('C13 LOTUS RETRIEVED',number+1,'/160',flush=True)
    frame = pd.DataFrame(records)
    assert not frame.empty
    frame.to_parquet(OUT / 'lotus-candidates.parquet',index=False)
    smiles = [s for s,v in cache.items() if v is not None]
    pd.DataFrame(dict(smiles=smiles,metric_key=[cache[s][0] for s in smiles])).to_parquet(OUT / 'fingerprint-smiles.parquet',index=False)
    np.save(OUT / 'fingerprints.npy',np.stack([cache[s][-1] for s in smiles]).astype(np.uint8))
    write(OUT / 'retrieval-freeze.json',dict(frozen_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=sha(OUT / 'protocol.json'),candidates_sha256=sha(OUT / 'lotus-candidates.parquet'),
        fingerprints_sha256=sha(OUT / 'fingerprints.npy'),index_sha256=sha(OUT / 'fingerprint-smiles.parquet'),
        query_labels_read=False,prediction_metrics_computed=False))
    # Only now read the already-consumed160 source labels. Never open C13 labels.
    labels = pd.read_csv(ROOT / 'outputs/local/c10-casmi2022-20261110/labels.csv',dtype=str).set_index('molecule_id')
    assert len(labels) == 160
    parent = old[old.is_parent.astype(bool)]
    def groups(data): return {mid:set(g.metric_key) for mid,g in data.groupby('molecule_id',sort=False)}
    pools = dict(parent=groups(parent),old_deep_pool=groups(old),lotus_all=groups(frame),lotus_cap128=groups(frame[frame.within_fixed_cap]))
    for name in ['lotus_all','lotus_cap128']:
        pools['parent_plus_'+name] = {mid:pools['parent'].get(mid,set())|pools[name].get(mid,set()) for mid in labels.index}
        pools['deep_plus_'+name] = {mid:pools['old_deep_pool'].get(mid,set())|pools[name].get(mid,set()) for mid in labels.index}
    coverage = {}
    for name,pool in pools.items():
        presence = [labels.loc[mid,'key'] in pool.get(mid,set()) for mid in labels.index]
        additions = [labels.loc[mid,'key'] not in pools['parent'].get(mid,set()) and labels.loc[mid,'key'] in pool.get(mid,set()) for mid in labels.index]
        sizes = [len(pool.get(mid,set())) for mid in labels.index]
        coverage[name] = dict(queries=160,truth_present=sum(presence),absent_parent_truths_recoverable=sum(additions),
            median_candidates=float(np.median(sizes)),maximum_candidates=max(sizes))
    write(OUT / 'coverage.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),
        development_queries=160,all_development_previously_used=True,coverage=coverage,
        retrieved_rows=len(frame),unique_smiles=len(smiles),counts=dict(counts),
        official_key_differs_from_mirror_raw_key=int(frame[['smiles','metric_key','provided_raw_key']].drop_duplicates().eval('metric_key != provided_raw_key').sum()),
        retrieval_freeze_sha256=sha(OUT / 'retrieval-freeze.json'),original_license='CC-BY-4.0',attribution_retained=True,
        fresh_C13_labels_read=False,model_scores_computed=False,recipe_selected=False,submission_qualified=False,
        limitation='Oracle presence on already-used positive-only development, not attainable rank/MRR or independent confirmation. A new recipe must freeze before fresh C13 metrics.'))
    print('DONE C13 USED-PANEL LOTUS COVERAGE',json.dumps(coverage),flush=True)


if __name__ == '__main__': main()
