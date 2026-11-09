"""Diagnostic/rejection check for one fixed public-policy external overlay."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import json
from functools import lru_cache
from datetime import datetime, timezone
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


def main():
    pa.set_cpu_count(2)
    assert rdBase.rdkitVersion == '2026.03.3'
    assert system_memory().get('free_gib', 0) >= 4
    out = ROOT / 'outputs/local/c09-external-reference-20261109'
    assert not (out / 'development.json').exists(), 'Fixed-policy diagnostic already sealed'
    policy_path = ROOT / 'configs/c09-external-reference-20261109.json'
    policy = json.loads(policy_path.read_text())
    freeze = json.loads((out / 'policy-freeze.json').read_text())
    assert sha(policy_path) == freeze['policy_sha256']
    selection = json.loads((ROOT / 'outputs/local/c09-followup-evaluation-20261109/selection.json').read_text())
    ids = selection['development_ids']; assert len(ids) == 30
    original = ROOT / 'outputs/forks/c08-external-validation'
    frame = pd.read_csv(original / 'c08_validation_p01.csv', dtype=str, keep_default_na=False).set_index('molecule_id').loc[ids].reset_index()
    queries_path = ROOT / 'outputs/local/c09-20261109/development/queries.parquet'
    queries = pd.read_parquet(queries_path)
    assert set(queries.molecule_id) == set(ids)
    references = ROOT / 'outputs/research/2026-11-09-silver/extlib'
    evidence, search_report = search(queries, references, features, policy)
    evidence.to_parquet(out / 'development-matches.parquet', index=False)
    library = pd.read_parquet(original / 'frozen_candidate_evidence.parquet', columns=['molecule_id', 'library_max'])
    confidence = library.groupby('molecule_id', sort=False).library_max.first().to_dict()
    labels_path = ROOT / 'outputs/publish/c08-validation-20261108/labels.csv'
    labels = pd.read_csv(labels_path, dtype=str).set_index('molecule_id').loc[ids]
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
            if mol is None:
                raise ValueError('Invalid external SMILES')
            Chem.RemoveStereochemistry(mol)
            identities[smiles] = Chem.MolToInchiKey(tautomer.Canonicalize(mol))[:14]
        return identities[smiles]
    before = metric_vector(frame, labels, identities); summaries = {}
    predictions = pd.DataFrame(dict(molecule_id=ids, P01=before))
    for view in ['all_reference', 'input_alias_excluded']:
        matches = evidence[evidence.view.eq(view)]
        final, changes = apply_overlay(frame, matches, confidence, policy, identity)
        after = metric_vector(final, labels, identities)
        final.to_csv(out / ('development-' + view + '.csv'), index=False)
        changes.to_csv(out / ('development-' + view + '-changes.csv'), index=False)
        summaries[view] = dict(paired=paired_summary(before, after), baseline_truths=int((before > 0).sum()),
                               candidate_truths=int((after > 0).sum()), actions=changes.reason.value_counts().to_dict(),
                               best_hits_matching_input_aliases=int(matches.get('input_alias', pd.Series(False, index=matches.index)).fillna(False).sum()))
        predictions[view] = after
    predictions.to_csv(out / 'development-predictions.csv', index=False)
    report = dict(campaign='C09E', completed_utc=datetime.now(timezone.utc).isoformat(), policy_sha256=sha(policy_path),
                  policy_freeze_sha256=sha(out / 'policy-freeze.json'), queries_sha256=sha(queries_path),
                  reference_sha256={name: sha(references / name) for name in ['ext_meta.parquet', 'ext_mz.npy', 'ext_in.npy']},
                  source_sha256={name: sha(ROOT / 'scripts' / name) for name in ['external_reference_overlay.py', 'run_c09_external_development.py', 'c09_spectrum_audit.py']},
                  search=search_report, development=summaries, official_rdkit=rdBase.rdkitVersion,
                  development_queries=30, confirmation_metrics_used=False, threshold_search=False,
                  development_rejects=bool(summaries['all_reference']['paired']['delta'] < 0), automatic_submission=False,
                  limitation='All development queries are previously used external-source records; signature exclusion is not full identity/library-provenance exclusion. Do not infer hidden improvement.')
    write(out / 'development.json', report)
    print('DONE C09 EXTERNAL DEVELOPMENT', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
