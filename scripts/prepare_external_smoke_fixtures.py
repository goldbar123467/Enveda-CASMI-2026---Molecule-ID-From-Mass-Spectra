"""Package real, already-used development inputs for changed-path Kaggle smoke.

Predicted answer lists and official structure identities are fixtures, not query
truth labels. They never enter the competition inference path or model choice.
"""
import json
import shutil
import pandas as pd
from precision_cache import ROOT, sha, write


def main():
    package = ROOT / 'outputs/publish/c09e-smoke-fixtures-20261109'; package.mkdir(parents=True, exist_ok=True)
    assert not (package / 'c09e-fixture-contract.json').exists(), 'Fixture package is sealed'
    out = ROOT / 'outputs/local/c09-external-reference-20261109'
    selection = json.loads((ROOT / 'outputs/local/c09-followup-evaluation-20261109/selection.json').read_text())
    ids = selection['development_ids']; assert len(ids) == 30
    source = ROOT / 'outputs/forks/c08-external-validation'
    baseline = pd.read_csv(source / 'c08_validation_p01.csv', dtype=str, keep_default_na=False).set_index('molecule_id').loc[ids].reset_index()
    baseline.to_csv(package / 'c09e_fixture_baseline.csv', index=False)
    queries = ROOT / 'outputs/local/c09-20261109/development/queries.parquet'
    frame = pd.read_parquet(queries)
    assert frame.molecule_id.nunique() == 30 and not {'key', 'smiles', 'inchikey14', 'formula'} & set(frame.columns)
    shutil.copy2(queries, package / 'c09e_fixture_queries.parquet')
    confidence = pd.read_parquet(source / 'frozen_candidate_evidence.parquet', columns=['molecule_id', 'library_max']).groupby('molecule_id', sort=False).library_max.first().reindex(ids)
    assert confidence.notna().all()
    confidence.reset_index().to_csv(package / 'c09e_fixture_library.csv', index=False)
    for src, dest in [('development-all_reference.csv', 'c09e_fixture_expected.csv'),
                      ('development-all_reference-changes.csv', 'c09e_fixture_expected_actions.csv'),
                      ('development-matches.parquet', 'c09e_fixture_expected_matches.parquet')]:
        shutil.copy2(out / src, package / dest)
    needed = set(s for text in baseline.smiles for s in text.split(';'))
    identities = []
    for name in ['neural_candidate_features.parquet', 'outsider_neural_features.parquet']:
        table = pd.read_parquet(source / name, columns=['smiles', 'metric_key']).dropna()
        identities.append(table[table.smiles.isin(needed)])
    identities = pd.concat(identities).drop_duplicates()
    assert not identities.smiles.duplicated().any() and set(identities.smiles) == needed
    identities.to_parquet(package / 'c09e_fixture_identities.parquet', index=False)
    names = sorted(p.name for p in package.iterdir() if p.is_file())
    development = json.loads((out / 'development.json').read_text())
    contract = dict(kind='real_input_changed_path_smoke_fixtures', queries=30, spectra=len(frame),
        query_truth_labels_included=False, source='Previously used30 development queries and actual complete P01 predictions',
        files={name: sha(package / name) for name in names}, reference_files=development['reference_sha256'],
        expected_changed_rows=7, expected_inserted_identities=2, expected_promoted_existing_identities=5,
        policy_sha256=development['policy_sha256'], development_evidence_sha256=sha(out / 'development.json'),
        purpose='Regression/execution path coverage only; no new coefficient choice or scientific evaluation')
    write(package / 'c09e-fixture-contract.json', contract)
    write(package / 'dataset-metadata.json', dict(id='clarkkitchen/casmi26-c09e-smoke-fixtures-20261109',
        title='CASMI26 C09E real input smoke fixtures 20261109', licenses=[dict(name='CC-BY-NC-4.0')]))
    print('SEALED REAL INPUT FIXTURES', json.dumps(contract), flush=True)


if __name__ == '__main__':
    main()
