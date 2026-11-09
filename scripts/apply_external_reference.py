"""Required P16 final stage: pinned P01 plus the sealed measured-reference rule."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import argparse
import hashlib
import importlib.util
import json
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
from rdkit import Chem, rdBase
from rdkit.Chem.MolStandardize import rdMolStandardize
from external_reference_overlay import search, apply_overlay


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--contract', type=Path, required=True)
    args = parser.parse_args(); work = args.work; code = args.contract.parent
    pa.set_cpu_count(2); assert rdBase.rdkitVersion == '2026.03.3'
    # Clear the scoring filename before any changed-stage work. A failed stage
    # cannot silently leave a baseline submission for Kaggle to score.
    original_path = work / 'c09e_p01_submission.csv'
    if not original_path.exists():
        original_path.write_bytes((work / 'submission.csv').read_bytes())
    (work / 'submission.csv').unlink(missing_ok=True)
    contract = json.loads(args.contract.read_text())
    for name, pin in contract['source_files'].items():
        assert sha(code / name) == pin, name
    policy_path = code / 'c09e_policy.json'; assert sha(policy_path) == contract['policy_sha256']
    policy = json.loads(policy_path.read_text())
    hits = list(args.input_root.rglob('ext_meta.parquet')); assert len(hits) == 1
    reference = hits[0].parent
    for name, pin in contract['reference_files'].items():
        assert sha(reference / name) == pin, name
    spec = importlib.util.spec_from_file_location('_c09e_features', code / 'c09e_features.py')
    features = importlib.util.module_from_spec(spec); spec.loader.exec_module(features)
    baseline = pd.read_csv(original_path, dtype=str, keep_default_na=False)
    query_path = work / 'effective_query_spectra.parquet'; queries = pd.read_parquet(query_path)
    assert queries.molecule_id.nunique() > 0 and baseline.molecule_id.is_unique
    matches, search_report = search(queries, reference, features, policy)
    matches.to_parquet(work / 'c09e_matches.parquet', index=False)
    evidence = pd.read_parquet(work / 'frozen_candidate_evidence.parquet', columns=['molecule_id', 'library_max'])
    confidence = evidence.groupby('molecule_id', sort=False).library_max.first().to_dict()
    identities = {}
    for name in ['neural_candidate_features.parquet', 'outsider_neural_features.parquet']:
        records = pd.read_parquet(work / name, columns=['smiles', 'metric_key']).dropna()
        for smiles, key in zip(records.smiles.astype(str), records.metric_key.astype(str)):
            assert smiles not in identities or identities[smiles] == key
            identities[smiles] = key
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
    final, actions = apply_overlay(baseline, matches[matches.view.eq('all_reference')], confidence, policy, identity)
    assert final.columns.tolist() == baseline.columns.tolist() == ['molecule_id', 'smiles']
    assert final.molecule_id.tolist() == baseline.molecule_id.tolist()
    for text in final.smiles:
        keys = [identity(s) for s in text.split(';')]
        assert 1 <= len(keys) <= 25 and len(keys) == len(set(keys))
    actions.to_csv(work / 'c09e_actions.csv', index=False)
    final.to_csv(work / 'submission.csv', index=False)
    receipt = dict(kind='P01_plus_fixed_external_reference', policy_sha256=contract['policy_sha256'],
        contract_sha256=sha(args.contract), reference_files=contract['reference_files'], source_files=contract['source_files'],
        baseline_sha256=sha(original_path), query_sha256=sha(query_path), csv_sha256=sha(work / 'submission.csv'),
        official_rdkit=rdBase.rdkitVersion, valid_rows=len(final), search=search_report,
        actions=actions.reason.value_counts().to_dict(), changed_rows=int(actions.changed.sum()),
        invalid_external_structures=int(actions.reason.eq('invalid_external_structure').sum()),
        official_identity_duplicates=0, required_stage_completed=True,
        development_evidence_sha256=contract['development_evidence_sha256'], scoring_submission=False)
    (work / 'c09e_external_receipt.json').write_text(json.dumps(receipt, indent=2))
    print('DONE C09E REQUIRED STAGE', json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
