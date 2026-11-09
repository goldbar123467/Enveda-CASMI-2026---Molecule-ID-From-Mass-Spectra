"""Report the already frozen C09E rule on complete80; never select or submit."""
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
from finish_c09_confirmation import CLOUD, metric_vector
from tail_fusion import paired_summary


def main():
    pa.set_cpu_count(2); assert rdBase.rdkitVersion == '2026.03.3'
    out = ROOT / 'outputs/local/c09-external-reference-20261109'
    assert not (out / 'confirmation.json').exists(), 'Confirmation already sealed'
    freeze = json.loads((out / 'implementation-freeze.json').read_text())
    for name, pin in freeze['source_files'].items():
        assert sha(ROOT / 'scripts' / name) == pin
    assert freeze['new80_metrics_read'] is False
    policy_path = ROOT / 'configs/c09-external-reference-20261109.json'
    policy = json.loads(policy_path.read_text()); assert sha(policy_path) == freeze['policy_sha256']
    original_report = json.loads((CLOUD / 'c09_clean_report.json').read_text())
    assert original_report['baseline_replay_exact'] and original_report['queries'] == 80
    labels = pd.read_csv(ROOT / 'outputs/publish/c09-validation-20261109/labels.csv', dtype=str).set_index('molecule_id')
    assert len(labels) == 80 and labels.partition.eq('confirmation').all()
    baseline = pd.read_csv(CLOUD / 'c09_clean_p01.csv', dtype=str, keep_default_na=False)
    queries = pd.read_parquet(CLOUD / 'c09_clean_queries.parquet')
    assert set(queries.molecule_id) == set(labels.index)
    references = ROOT / 'outputs/research/2026-11-09-silver/extlib'
    for name, pin in freeze['reference_files'].items():
        assert sha(references / name) == pin
    matches, search_report = search(queries, references, features, policy)
    matches.to_parquet(out / 'confirmation-matches.parquet', index=False)
    identities = {}; confidence = None
    for name in ['c09_clean_neural_p01.parquet', 'outsider_neural_features.parquet']:
        records = pd.read_parquet(CLOUD / name)
        if name == 'c09_clean_neural_p01.parquet':
            # The old saved pipeline shows exact equality to frozen evidence.
            # Per-query library_max is constant and identifies its protection gate.
            assert records.groupby('molecule_id').library_max.nunique().max() == 1
            confidence = records.groupby('molecule_id').library_max.first().to_dict()
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
    before = metric_vector(baseline, labels, identities)
    negative = pd.read_csv(CLOUD / 'c09_clean_predictions.csv').set_index('molecule_id').loc[labels.index]
    assert np.allclose(before, negative.p01_mrr25, rtol=0, atol=1e-14)
    raw_matches = pd.read_csv(ROOT / 'outputs/forks/c09-spectrum-audit-v3/c09_spectrum_matches.csv')
    raw_clean = ~labels.index.isin(set(raw_matches.molecule_id))
    values = {'P01': before, 'P14': negative.p14_mrr25.to_numpy()}
    actions_by_view = {}
    for view in ['all_reference', 'input_alias_excluded']:
        final, actions = apply_overlay(baseline, matches[matches.view.eq(view)], confidence, policy, identity)
        values[view] = metric_vector(final, labels, identities)
        final.to_csv(out / ('confirmation-' + view + '.csv'), index=False)
        actions.to_csv(out / ('confirmation-' + view + '-changes.csv'), index=False)
        actions_by_view[view] = actions.reason.value_counts().to_dict()
    cohorts = {}
    for name, mask in [('full80', np.ones(80, dtype=bool)), ('no_raw_competition_signature_overlap', raw_clean)]:
        cohorts[name] = dict(queries=int(mask.sum()), excluded=int((~mask).sum()),
            sources=labels.loc[mask, 'source'].value_counts().to_dict(),
            baseline_truths=int((before[mask] > 0).sum()),
            candidate_truths={view: int((values[view][mask] > 0).sum()) for view in ['all_reference', 'input_alias_excluded']},
            comparisons={view: paired_summary(before[mask], value[mask]) for view, value in values.items() if view != 'P01'})
    report = dict(campaign='C09E', completed_utc=datetime.now(timezone.utc).isoformat(),
        policy_sha256=sha(policy_path), implementation_freeze_sha256=sha(out / 'implementation-freeze.json'),
        complete_baseline_report_sha256=sha(CLOUD / 'c09_clean_report.json'), search=search_report,
        cohorts=cohorts, actions=actions_by_view,
        detects_known_P14_regression=bool((values['P14'] - before).mean() < 0),
        truth_absent_mrr=0, confirmation_recipe_selection=False, automatic_submission=False,
        broader_pretraining_exposure_excluded=False,
        limitation='External-source confirmation, signature exclusions only; missing per-reference acquisition/provenance attribution prevents universal leakage exclusion. Assess controls and actual scored reproduction.')
    pd.DataFrame(dict(molecule_id=labels.index, **values)).to_csv(out / 'confirmation-predictions.csv', index=False)
    write(out / 'confirmation.json', report)
    print('DONE C09E CONFIRMATION', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
