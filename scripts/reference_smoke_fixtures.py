"""Run the changed overlay paths on pinned real inputs, separate from scoring."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import argparse
import hashlib
import importlib.util
import json
import shutil
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem.MolStandardize import rdMolStandardize
from external_reference_overlay import search, apply_overlay


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True); parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--code', type=Path, required=True); parser.add_argument('--fixture-pin', required=True)
    args = parser.parse_args(); assert rdBase.rdkitVersion == '2026.03.3'
    hits = list(args.input_root.rglob('c09e-fixture-contract.json')); assert len(hits) == 1 and sha(hits[0]) == args.fixture_pin
    package = hits[0].parent; contract = json.loads(hits[0].read_text())
    assert contract['queries'] == 30 and not contract['query_truth_labels_included']
    for name, expected in contract['files'].items():
        assert sha(package / name) == expected
    reference_hits = list(args.input_root.rglob('ext_meta.parquet')); assert len(reference_hits) == 1
    reference = reference_hits[0].parent
    for name, expected in contract['reference_files'].items():
        assert sha(reference / name) == expected
    policy_path = args.code / 'c09e_policy.json'; assert sha(policy_path) == contract['policy_sha256']
    policy = json.loads(policy_path.read_text())
    feature_spec = importlib.util.spec_from_file_location('_c09e_fixture_features', args.code / 'c09e_features.py')
    features = importlib.util.module_from_spec(feature_spec); feature_spec.loader.exec_module(features)
    queries = pd.read_parquet(package / 'c09e_fixture_queries.parquet')
    assert not {'key', 'smiles', 'inchikey14', 'formula'} & set(queries.columns)
    baseline = pd.read_csv(package / 'c09e_fixture_baseline.csv', dtype=str, keep_default_na=False)
    expected = pd.read_csv(package / 'c09e_fixture_expected.csv', dtype=str, keep_default_na=False)
    confidence = pd.read_csv(package / 'c09e_fixture_library.csv').set_index('molecule_id').library_max.to_dict()
    table = pd.read_parquet(package / 'c09e_fixture_identities.parquet')
    assert table.smiles.is_unique
    identities = dict(zip(table.smiles.astype(str), table.metric_key.astype(str)))
    tautomer = rdMolStandardize.TautomerEnumerator()
    @lru_cache(maxsize=30000)
    def identity(smiles):
        if smiles not in identities:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError('Invalid external structure')
            Chem.RemoveStereochemistry(mol)
            identities[smiles] = Chem.MolToInchiKey(tautomer.Canonicalize(mol))[:14]
        return identities[smiles]
    matches, search_report = search(queries, reference, features, policy)
    final, actions = apply_overlay(baseline, matches[matches.view.eq('all_reference')], confidence, policy, identity)
    assert final.equals(expected), 'Pinned real-input fixture output differs'
    assert int(actions.changed.sum()) == contract['expected_changed_rows'] == 7
    counts = actions.reason.value_counts().to_dict()
    assert counts.get('inserted_external_identity') == 2 and counts.get('promoted_existing_identity') == 5
    for text in final.smiles:
        keys = [identity(s) for s in text.split(';')]; assert 1 <= len(keys) <= 25 and len(keys) == len(set(keys))
    expected_matches = pd.read_parquet(package / 'c09e_fixture_expected_matches.parquet')
    columns = ['molecule_id', 'view']; left = matches.set_index(columns).sort_index(); right = expected_matches.set_index(columns).reindex(left.index)
    error = float(np.abs(left.cosine.to_numpy() - right.cosine.to_numpy()).max())
    assert error < 1e-12 and left.matched_peaks.equals(right.matched_peaks) and left.smiles.equals(right.smiles)
    # Explicit invalid-structure fallback runs on a real fixture baseline row.
    invalid = pd.DataFrame(dict(molecule_id=[str(baseline.molecule_id.iloc[0])], cosine=[.99], matched_peaks=[8], smiles=['INVALID_SMILES']))
    def invalid_identity(value):
        if value == 'INVALID_SMILES':
            raise ValueError('Intentional invalid-structure fixture')
        return identity(value)
    fallback, fallback_actions = apply_overlay(baseline.iloc[:1], invalid, {str(baseline.molecule_id.iloc[0]): 0.}, policy, invalid_identity)
    assert fallback.equals(baseline.iloc[:1]) and fallback_actions.reason.iloc[0] == 'invalid_external_structure'
    for name in ['c09e_fixture_baseline.csv', 'c09e_fixture_queries.parquet', 'c09e_fixture_library.csv', 'c09e_fixture_identities.parquet']:
        shutil.copy2(package / name, args.work / name)
    matches.to_parquet(args.work / 'c09e_fixture_matches.parquet', index=False)
    final.to_csv(args.work / 'c09e_fixture_final.csv', index=False); actions.to_csv(args.work / 'c09e_fixture_actions.csv', index=False)
    receipt = dict(real_queries=30, changed_rows=7, inserted_identities=2, promoted_existing_identities=5,
        fixture_contract_sha256=args.fixture_pin, query_truth_labels_used=False, scoring_csv_modified=False,
        exact_expected_lists=True, local_cloud_peak_score_max_abs_difference=error, official_identity_duplicates=0,
        invalid_structure_fallback_passed=True, actions=counts, search=search_report)
    (args.work / 'p16_fixture_receipt.json').write_text(json.dumps(receipt, indent=2))
    print('P16 REAL CHANGED-PATH FIXTURES COMPLETE', json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
