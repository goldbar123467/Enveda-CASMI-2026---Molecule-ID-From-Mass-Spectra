"""One reserved120 evaluation of the development-frozen C10 winner; never submit."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import argparse
import json
from datetime import datetime, timezone
from functools import lru_cache

import numpy as np
import pandas as pd
import pyarrow as pa
from rdkit import Chem, rdBase
from rdkit.Chem.MolStandardize import rdMolStandardize

from precision_cache import ROOT, sha, write
from c10_reference_search import OUT, POOL, CONFIG, POLICY
from develop_c10_candidates import CLOUD, LOCAL, select_recipe
from external_reference_overlay import apply_overlay
from finish_c09_confirmation import metric_vector
from tail_fusion import paired_summary


def statistical_gate(comparisons):
    """Full incremental support against both controls, plus alias sensitivity."""
    failures = []
    for name in ['winner_vs_P01', 'winner_vs_P16', 'alias_excluded_winner_vs_P01', 'alias_excluded_winner_vs_P16']:
        summary = comparisons[name]
        assert summary['n'] == 120
        assert np.isfinite([summary['delta'], *summary['ci95']]).all()
        alias = name.startswith('alias_excluded')
        supported = (summary['delta'] >= 0 and summary['ci95'][0] >= 0) if alias else (summary['delta'] > 0 and summary['ci95'][0] > 0)
        if not supported: failures.append(name)
    return dict(passed=not failures, failed_comparisons=failures)


def prepare_protocol():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'confirmation-protocol.json'
    assert not path.exists() and not (OUT / 'confirmation.json').exists(), 'Protocol/results already sealed'
    assert not (OUT / 'development.json').exists(), 'Prepare prospectively before development results'
    sources = ['confirm_c10_candidate.py', 'develop_c10_candidates.py', 'c10_reference_search.py',
        'external_reference_overlay.py', 'finish_c09_confirmation.py', 'tail_fusion.py']
    protocol = dict(prepared_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256={name: sha(ROOT / 'scripts' / name) for name in sources},
        config_sha256=sha(CONFIG), policy_sha256=sha(POLICY),
        development_protocol_sha256=sha(OUT / 'development-protocol.json'),
        reference_manifest_sha256=sha(POOL / 'reference-manifest.json'),
        local_labels_sha256=sha(LOCAL / 'labels.csv'), partitions_sha256=sha(LOCAL / 'partitions.json'),
        validation_contract_sha256=sha(ROOT / 'outputs/publish/c10-casmi2022-20261110/validation-contract.json'),
        selection='Use only already-frozen new winner; do not evaluate unselected variants or reselect',
        gate='Reserved120 positive paired95% lower bound versus both P01/P16; alias-excluded delta/lower bound >=0 versus both',
        controls='Report P14/P15 deltas and whether known public regressions are reproduced, even if they contradict the panel',
        missing_truth_mrr=0, bootstrap_draws=10000, bootstrap_seed=20261105,
        development_metrics_read=False, reserved_labels_read=False, reserved_metrics_computed=False,
        automatic_submission=False, full_submission_qualification='Still requires source/provenance review, real Kaggle changed-path smoke and independent execution/parity audit')
    write(path, protocol)
    print('DONE C10 CONFIRMATION PROTOCOL', sha(path), flush=True)


def evaluate():
    pa.set_cpu_count(2); assert rdBase.rdkitVersion == '2026.03.3'
    assert not (OUT / 'confirmation.json').exists(), 'Reserved result already sealed'
    assert not (OUT / 'confirmation-input-freeze.json').exists(), 'Reserved evaluation already started; inspect existing artifacts before recovery'
    protocol = json.loads((OUT / 'confirmation-protocol.json').read_text())
    for name, pin in protocol['source_sha256'].items(): assert sha(ROOT / 'scripts' / name) == pin, name
    for path, key in [(CONFIG, 'config_sha256'), (POLICY, 'policy_sha256'),
        (OUT / 'development-protocol.json', 'development_protocol_sha256'),
        (POOL / 'reference-manifest.json', 'reference_manifest_sha256'),
        (LOCAL / 'labels.csv', 'local_labels_sha256'), (LOCAL / 'partitions.json', 'partitions_sha256')]:
        assert sha(path) == protocol[key], key
    freeze = json.loads((OUT / 'selection-freeze.json').read_text())
    assert freeze['new_candidate_selected'], 'P16 won/tied development; leave reserved labels untouched'
    assert not freeze['reserved_labels_read'] and not freeze['reserved_metrics_computed']
    assert freeze['development_sha256'] == sha(OUT / 'development.json')
    assert freeze['development_protocol_sha256'] == protocol['development_protocol_sha256']
    assert freeze['config_sha256'] == protocol['config_sha256'] and freeze['policy_sha256'] == protocol['policy_sha256']
    assert freeze['reference_manifest_sha256'] == protocol['reference_manifest_sha256']
    assert freeze['matches_sha256'] == sha(OUT / 'matches.parquet')
    for name, pin in freeze['source_sha256'].items(): assert sha(ROOT / 'scripts' / name) == pin, name
    development = json.loads((OUT / 'development.json').read_text())
    recipes = json.loads(CONFIG.read_text())['candidates_in_tie_order']
    scores = {r['name']: development['candidates'][r['name'] + ':all_reference']['comparison']['candidate_mrr25'] for r in recipes}
    assert freeze['winner'] == development['winner'] == select_recipe(scores, recipes)
    winner = freeze['winner']['name']; assert winner != 'P16'
    cloud_report = json.loads((CLOUD / 'c10_report.json').read_text())
    assert sha(CLOUD / 'c10_report.json') == freeze['cloud_report_sha256']
    assert cloud_report['baseline_replay_exact'] and cloud_report['queries'] == 160
    assert cloud_report['validation_contract_sha256'] == protocol['validation_contract_sha256']
    assert not cloud_report['metrics_computed'] and not cloud_report['query_labels_mounted']
    assert sha(CLOUD / 'c10_p01.csv') == cloud_report['baseline_sha256']
    p15_inference = json.loads((OUT / 'p15-control-inference.json').read_text())
    assert not p15_inference['query_labels_used'] and not p15_inference['metrics_computed']
    assert p15_inference['receipt']['input_errors'] == 0 and p15_inference['receipt']['scored_molecules'] == 160
    # Seal all prediction inputs before opening the reserved labels.
    inputs = [OUT / 'selection-freeze.json', OUT / 'development.json', OUT / 'matches.parquet',
        OUT / 'p15-control.csv', OUT / 'p15-control-inference.json', CLOUD / 'c10_p01.csv',
        CLOUD / 'c10_p14.csv', CLOUD / 'c10_neural_p01.parquet', CLOUD / 'outsider_neural_features.parquet']
    write(OUT / 'confirmation-input-freeze.json', dict(frozen_utc=datetime.now(timezone.utc).isoformat(),
        winner=freeze['winner'], confirmation_protocol_sha256=sha(OUT / 'confirmation-protocol.json'),
        input_sha256={str(path.relative_to(ROOT)).replace('\\', '/'): sha(path) for path in inputs},
        reserved_labels_read=False, reserved_metrics_computed=False))
    labels = pd.read_csv(LOCAL / 'labels.csv', dtype=str, skiprows=range(1, 41)).set_index('molecule_id')
    partitions = json.loads((LOCAL / 'partitions.json').read_text())
    assert len(labels) == 120 and labels.partition.eq('confirmation').all()
    assert labels.index.tolist() == partitions['confirmation'] and not set(labels.index) & set(partitions['development'])
    identities = {}; confidence = None
    for name in ['c10_neural_p01.parquet', 'outsider_neural_features.parquet']:
        records = pd.read_parquet(CLOUD / name)
        if name == 'c10_neural_p01.parquet':
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
            if mol is None: raise ValueError('Invalid external structure')
            Chem.RemoveStereochemistry(mol)
            identities[smiles] = Chem.MolToInchiKey(tautomer.Canonicalize(mol))[:14]
        return identities[smiles]
    def subset(frame): return frame.set_index('molecule_id').loc[labels.index].reset_index()
    baseline = subset(pd.read_csv(CLOUD / 'c10_p01.csv', dtype=str, keep_default_na=False))
    values = {'P01': metric_vector(baseline, labels, identities)}
    for name, path in [('P14', CLOUD / 'c10_p14.csv'), ('P15', OUT / 'p15-control.csv')]:
        values[name] = metric_vector(subset(pd.read_csv(path, dtype=str, keep_default_na=False)), labels, identities)
    matches = pd.read_parquet(OUT / 'matches.parquet'); policy = json.loads(POLICY.read_text())
    actions_by_view = {}
    for name in ['P16', winner]:
        for view in ['all_reference', 'input_alias_excluded']:
            evidence = matches[matches.recipe.eq(name) & matches.view.eq(view)]
            final, actions = apply_overlay(baseline, evidence, confidence, policy, identity)
            values[name + ':' + view] = metric_vector(final, labels, identities)
            final.to_csv(OUT / ('confirmation-' + name + '-' + view + '.csv'), index=False)
            actions.to_csv(OUT / ('confirmation-' + name + '-' + view + '-actions.csv'), index=False)
            actions_by_view[name + ':' + view] = actions.reason.value_counts().to_dict()
    comparisons = {}
    for view, prefix in [('all_reference', ''), ('input_alias_excluded', 'alias_excluded_')]:
        for control in ['P01', 'P16']:
            before = values['P01'] if control == 'P01' else values['P16:' + view]
            comparisons[prefix + 'winner_vs_' + control] = paired_summary(before, values[winner + ':' + view])
    gate = statistical_gate(comparisons)
    controls = {name: paired_summary(values['P01'], values[name]) for name in ['P14', 'P15']}
    report = dict(completed_utc=datetime.now(timezone.utc).isoformat(), winner=freeze['winner'], queries=120,
        sources=labels.source.value_counts().to_dict(), comparisons=comparisons, statistical_gate=gate,
        baseline_mrr25=float(values['P01'].mean()), p16_mrr25=float(values['P16:all_reference'].mean()),
        truth_coverage={name: int((value > 0).sum()) for name, value in values.items()},
        actions=actions_by_view, controls=controls,
        detects_known_public_regression={name: summary['delta'] < 0 for name, summary in controls.items()},
        negative_control_interpretation='Report both control signs without filtering molecules. Failure to reproduce public regressions limits hidden-score calibration even if the incremental gate passes.',
        truth_absent_mrr=0, reserved_recipe_reselection=False,
        selection_freeze_sha256=sha(OUT / 'selection-freeze.json'),
        confirmation_protocol_sha256=sha(OUT / 'confirmation-protocol.json'),
        input_freeze_sha256=sha(OUT / 'confirmation-input-freeze.json'),
        broader_pretraining_exposure_excluded=False, automatic_submission=False, submission_qualified=False,
        next_action='If the statistical gate passes, complete source/rights review plus real changed-path Kaggle smoke and independent parity audit. Wait for actual P16 result and retain one pending submission globally.',
        limitation='CASMI2022 external-source panel, single selected LC-MS2 acquisition, unavailable collision energy, incomplete parent reference provenance and possible historical pretraining exposure.')
    pd.DataFrame(dict(molecule_id=labels.index, **values)).to_csv(OUT / 'confirmation-predictions.csv', index=False)
    write(OUT / 'confirmation.json', report)
    print('DONE C10 FROZEN CONFIRMATION', json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare-protocol', action='store_true')
    group.add_argument('--evaluate', action='store_true')
    arguments = parser.parse_args()
    prepare_protocol() if arguments.prepare_protocol else evaluate()
