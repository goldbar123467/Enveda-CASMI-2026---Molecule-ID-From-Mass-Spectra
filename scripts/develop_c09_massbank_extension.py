"""Test one prospective reference extension on old30 development only.

The original compilation is CC-BY-NC-SA-4.0. Preserve record attribution and
literal record-license fields; private research only, no P16 changes or upload.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import json
from datetime import datetime, timezone
from functools import lru_cache

import numpy as np
import pandas as pd
import pyarrow as pa
from rdkit import Chem, rdBase
from rdkit.Chem.MolStandardize import rdMolStandardize

from precision_cache import ROOT, sha, write, features
from external_reference_overlay import search, apply_overlay
from finish_c09_confirmation import metric_vector
from tail_fusion import paired_summary
from training_monitor import system_memory

OUT = ROOT / 'outputs/local/c09-massbank-extension-20261110'
P16 = ROOT / 'outputs/local/c09-external-reference-20261109'


def unread_confirmation():
    assert not (P16 / 'confirmation.json').exists(), 'New80 confirmation already available'
    assert not (ROOT / 'outputs/local/c09-followup-evaluation-20261109/confirmation.json').exists()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def main():
    pa.set_cpu_count(2); unread_confirmation()
    assert rdBase.rdkitVersion == '2026.03.3' and system_memory()['free_gib'] >= 4
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'development.json').exists(), 'Development experiment already sealed'
    policy_path = ROOT / 'configs/c09-external-reference-20261109.json'
    policy = json.loads(policy_path.read_text())
    parent = json.loads((P16 / 'development.json').read_text())
    assert sha(policy_path) == parent['policy_sha256']
    for name, expected in parent['source_sha256'].items():
        assert sha(ROOT / 'scripts' / name) == expected, name
    source = ROOT / 'outputs/research/2026-11-09-silver/massbank-coverage-audit/spectra.parquet'
    write(OUT / 'protocol.json', dict(prepared_utc=stamp(), new80_metrics_read=False,
        policy_sha256=sha(policy_path), parent_development_sha256=sha(P16 / 'development.json'),
        parent_matches_sha256=sha(P16 / 'development-matches.parquet'),
        source_sha256=sha(source), experiment_sha256=sha(__file__),
        selection='One literalCC0 positiveMS2 >=4peak extension; same fixed thresholds, no tuning. Retain only if old30 MRR increases versus P16 with zero harmed queries.',
        future_confirmation='A retained extension must separately demonstrate incremental positive95% lower-bound gain versus P16 in full80 and raw-clean60 before any trial. No confirmation recipe reselection.',
        compilation_license='CC-BY-NC-SA-4.0; retain source/record attribution; no public redistribution or P16 adoption',
        automatic_submission=False))
    raw = pd.read_parquet(source, filters=[('license', '=', 'CC0'), ('ion_mode', '=', 'POSITIVE'), ('ms_type', '=', 'MS2')])
    raw = raw[raw.num_peaks.ge(4) & np.isfinite(raw.precursor_mz) & raw.precursor_mz.gt(0)].copy()
    rows, mzs, intensities = [], [], []
    rejected = dict(peak_shape=0, fewer_than_four_positive_peaks=0, invalid_structure_or_key=0)
    for row in raw.itertuples(index=False):
        mz, intensity = np.asarray(row.mzs, dtype=np.float32), np.asarray(row.intensities, dtype=np.float32)
        if mz.ndim != 1 or mz.shape != intensity.shape:
            rejected['peak_shape'] += 1; continue
        keep = np.isfinite(mz) & np.isfinite(intensity) & (mz > 0) & (intensity > 0)
        mz, intensity = mz[keep], intensity[keep]
        if len(mz) < 4:
            rejected['fewer_than_four_positive_peaks'] += 1; continue
        mol = Chem.MolFromSmiles(str(row.smiles))
        if mol is None or Chem.MolToInchiKey(mol)[:14] != str(row.inchikey)[:14]:
            rejected['invalid_structure_or_key'] += 1; continue
        chosen = np.argsort(-intensity, kind='stable')[:128]
        chosen = chosen[np.argsort(mz[chosen], kind='stable')]
        a, b = np.zeros(128, np.float32), np.zeros(128, np.float32)
        a[:len(chosen)], b[:len(chosen)] = mz[chosen], intensity[chosen]
        metadata = row._asdict()
        metadata.pop('mzs'); metadata.pop('intensities')
        metadata.update(inchikey14=str(row.inchikey)[:14], normalized_smiles=Chem.MolToSmiles(mol),
                        ionization_mode='positive', parent_compilation_license='CC-BY-NC-SA-4.0')
        rows.append(metadata); mzs.append(a); intensities.append(b)
    references = OUT / 'references'; references.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_parquet(references / 'ext_meta.parquet', index=False)
    np.save(references / 'ext_mz.npy', np.asarray(mzs)); np.save(references / 'ext_in.npy', np.asarray(intensities))
    ids = json.loads((ROOT / 'outputs/local/c09-followup-evaluation-20261109/selection.json').read_text())['development_ids']
    assert len(ids) == 30
    original = ROOT / 'outputs/forks/c08-external-validation'
    frame = pd.read_csv(original / 'c08_validation_p01.csv', dtype=str, keep_default_na=False).set_index('molecule_id').loc[ids].reset_index()
    queries_path = ROOT / 'outputs/local/c09-20261109/development/queries.parquet'
    assert sha(queries_path) == parent['queries_sha256']
    queries = pd.read_parquet(queries_path); assert set(queries.molecule_id) == set(ids)
    added, search_report = search(queries, references, features, policy)
    added.to_parquet(OUT / 'supplemental-matches.parquet', index=False)
    combined = pd.read_parquet(P16 / 'development-matches.parquet').set_index(['molecule_id', 'view'])
    supplemental = added.set_index(['molecule_id', 'view']).reindex(index=combined.index, columns=combined.columns)
    present = supplemental.reference_row.notna()
    supplemental.loc[present, 'reference_row'] += parent['search']['reference_rows']
    # Strictly larger similarity reproduces concatenation with P16 references
    # first. Equal-score ties preserve the original fixed-reference result.
    replace = supplemental.cosine > combined.cosine
    combined.loc[replace, :] = supplemental.loc[replace, combined.columns]
    combined = combined.reset_index(); combined.to_parquet(OUT / 'combined-matches.parquet', index=False)
    library = pd.read_parquet(original / 'frozen_candidate_evidence.parquet', columns=['molecule_id', 'library_max'])
    confidence = library.groupby('molecule_id', sort=False).library_max.first().to_dict()
    labels = pd.read_csv(ROOT / 'outputs/publish/c08-validation-20261108/labels.csv', dtype=str).set_index('molecule_id').loc[ids]
    assert labels.partition.eq('development').all()
    identities = {}
    for name in ['neural_candidate_features.parquet', 'outsider_neural_features.parquet']:
        records = pd.read_parquet(original / name, columns=['smiles', 'metric_key']).dropna()
        for smiles, key in zip(records.smiles.astype(str), records.metric_key.astype(str)):
            assert smiles not in identities or identities[smiles] == key
            identities[smiles] = key
    tautomer = rdMolStandardize.TautomerEnumerator()
    @lru_cache(maxsize=30000)
    def identity(smiles):
        if smiles not in identities:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None: raise ValueError('Invalid external structure')
            Chem.RemoveStereochemistry(mol)
            identities[smiles] = Chem.MolToInchiKey(tautomer.Canonicalize(mol))[:14]
        return identities[smiles]
    previous = pd.read_csv(P16 / 'development-predictions.csv').set_index('molecule_id').loc[ids]
    before = metric_vector(frame, labels, identities)
    assert np.allclose(before, previous.P01, rtol=0, atol=1e-14)
    summaries = {}; predictions = pd.DataFrame(dict(molecule_id=ids, P01=before))
    for view in ['all_reference', 'input_alias_excluded']:
        final, changes = apply_overlay(frame, combined[combined.view.eq(view)], confidence, policy, identity)
        after = metric_vector(final, labels, identities)
        final.to_csv(OUT / ('development-' + view + '.csv'), index=False)
        changes.to_csv(OUT / ('changes-' + view + '.csv'), index=False)
        summaries[view] = dict(versus_P01=paired_summary(before, after),
                              versus_P16=paired_summary(previous[view].to_numpy(), after))
        predictions[view] = after
    predictions.to_csv(OUT / 'development-predictions.csv', index=False)
    unread_confirmation()
    primary = summaries['all_reference']['versus_P16']
    report = dict(completed_utc=stamp(), new80_metrics_read=False, development=summaries,
        supplemental_records=len(rows), rejected=rejected, search=search_report,
        changed_best_matches=int(replace.sum()), retained=bool(primary['delta'] > 1e-12 and primary['harmed'] == 0),
        policy_sha256=sha(policy_path), protocol_sha256=sha(OUT / 'protocol.json'),
        source_sha256=sha(__file__), reference_sha256={name: sha(references / name) for name in ['ext_meta.parquet', 'ext_mz.npy', 'ext_in.npy']},
        threshold_search=False, original_P16_unchanged=True, automatic_submission=False,
        limitation='Used development data, raw source license metadata retained; no hidden gain, universal acquisition exclusion or additional redistribution rights inferred.')
    write(OUT / 'development.json', report)
    print('DONE C09 MASSBANK EXTENSION', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
