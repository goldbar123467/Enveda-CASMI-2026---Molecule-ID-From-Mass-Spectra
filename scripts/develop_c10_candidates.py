"""Use only C10's40 development labels; freeze selection before reserved metrics."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import json
from datetime import datetime, timezone
from functools import lru_cache

import numpy as np
import pandas as pd
import pyarrow as pa
import torch
from rdkit import Chem, rdBase
from rdkit.Chem.MolStandardize import rdMolStandardize

from precision_cache import ROOT, sha, write, features
from c10_reference_search import OUT, POOL, CONFIG, POLICY
from external_reference_overlay import apply_overlay
from finish_c09_confirmation import metric_vector
from followup_models import score_group, PACKAGE
from compatible_tail import apply
from tail_fusion import paired_summary
from training_monitor import system_memory

CLOUD = ROOT / 'outputs/forks/c10-casmi2022-validation-v1'
LOCAL = ROOT / 'outputs/local/c10-casmi2022-20261110'


def select_recipe(scores, recipes):
    """Numerical ties preserve the predeclared order, starting with P16."""
    maximum = max(scores.values())
    return next(recipe for recipe in recipes if scores[recipe['name']] >= maximum - 1e-12)


def main():
    pa.set_cpu_count(2); torch.set_num_threads(2)
    assert rdBase.rdkitVersion == '2026.03.3' and system_memory()['free_gib'] >= 4
    assert not (OUT / 'development.json').exists(), 'Completed development is sealed'
    protocol = json.loads((OUT / 'development-protocol.json').read_text())
    for name, pin in protocol['source_sha256'].items(): assert sha(ROOT / 'scripts' / name) == pin, name
    assert sha(CONFIG) == protocol['config_sha256']
    assert sha(POLICY) == protocol['policy_sha256']
    assert sha(PACKAGE / 'followup-manifest.json') == protocol['p15_manifest_sha256']
    assert sha(LOCAL / 'labels.csv') == protocol['local_labels_sha256']
    assert sha(POOL / 'reference-manifest.json') == protocol['reference_manifest_sha256']
    collector = json.loads((LOCAL / 'collector-status.json').read_text())
    assert collector['state'] == 'collected' and not collector['label_metrics_computed']
    report = json.loads((CLOUD / 'c10_report.json').read_text())
    assert report['queries'] == 160 and report['baseline_replay_exact'] and not report['metrics_computed']
    assert not report['query_labels_mounted'] and not report['recipe_selection']
    assert report['validation_contract_sha256'] == protocol['validation_contract_sha256']
    devices = json.loads((CLOUD / 'c10_t4_allocation.json').read_text())['devices']
    assert len(devices) == 2 and all('T4' in d['name'] for d in devices)
    preflight = json.loads((CLOUD / 'c10_gpu1_preflight.json').read_text())
    assert preflight['dreams_finite'] and preflight['molformer_finite']
    assert preflight['physical_device_requested'] == '1'
    search_report = json.loads((OUT / 'search-report.json').read_text())
    assert search_report['parent_exact_parity'] and not search_report['metrics_computed']
    assert sha(OUT / 'matches.parquet') == search_report['matches_sha256']
    baseline = pd.read_csv(CLOUD / 'c10_p01.csv', dtype=str, keep_default_na=False)
    assert sha(CLOUD / 'c10_p01.csv') == report['baseline_sha256']
    assert len(baseline) == 160 and baseline.molecule_id.is_unique
    local_queries = pd.read_parquet(ROOT / 'outputs/publish/c10-casmi2022-20261110/validation-queries.parquet')
    assert set(local_queries.molecule_id) == set(baseline.molecule_id)
    cloud_queries = pd.read_parquet(CLOUD / 'c10_queries.parquet').set_index('molecule_id')
    assert cloud_queries.index.is_unique and set(cloud_queries.index) == set(local_queries.molecule_id)
    for row in local_queries.itertuples(index=False):
        cloud = cloud_queries.loc[row.molecule_id]
        assert float(row.precursor_mz) == float(cloud.precursor_mz) and row.adduct == cloud.adduct
        assert row.ionization_mode == cloud.ionization_mode
        for name in ['ms2_mzs', 'ms2_normalized_intensities']:
            assert np.array_equal(features.array_value(getattr(row, name)), features.array_value(cloud[name])), 'Query search/serving input mismatch'
    # The RTX2070 performs a useful, label-blind negative-control inference once.
    # This is the previously failed P15 serving recipe, not another training fit.
    torch.cuda.set_per_process_memory_fraction(.7)
    scores, p15_receipt = score_group('p15-parent', baseline, CLOUD / 'c10_queries.parquet',
        OUT / 'p15-control-scores.parquet', {}, OUT / 'bundles')
    assert p15_receipt['input_errors'] == 0 and p15_receipt['scored_molecules'] == 160
    p15, changes = apply(baseline, scores, .2)
    assert changes['rank_one_changes'] == 0 and changes['candidate_set_changes'] == 0
    assert changes['passthrough_molecules'] == 0
    p15.to_csv(OUT / 'p15-control.csv', index=False)
    write(OUT / 'p15-control-inference.json', dict(completed_utc=datetime.now(timezone.utc).isoformat(),
        receipt=p15_receipt, changes=changes, query_labels_used=False, metrics_computed=False))
    print('C10 P15 CONTROL INFERENCE COMPLETE160', flush=True)
    # Only the first40 preassigned development labels are read. The120 reserved
    # labels and their metrics remain unused in this program.
    labels = pd.read_csv(LOCAL / 'labels.csv', dtype=str, nrows=40).set_index('molecule_id')
    partitions = json.loads((LOCAL / 'partitions.json').read_text())
    assert len(labels) == 40 and labels.partition.eq('development').all()
    assert labels.index.tolist() == partitions['development']
    assert not set(labels.index) & set(partitions['confirmation'])
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
    baseline = subset(baseline)
    values = {'P01': metric_vector(baseline, labels, identities)}
    for name, frame in [('P14', pd.read_csv(CLOUD / 'c10_p14.csv', dtype=str)), ('P15', p15)]:
        values[name] = metric_vector(subset(frame), labels, identities)
    matches = pd.read_parquet(OUT / 'matches.parquet')
    policy = json.loads(POLICY.read_text()); recipes = json.loads(CONFIG.read_text())['candidates_in_tie_order']
    summaries = {}
    for recipe in recipes:
        for view in ['all_reference', 'input_alias_excluded']:
            name = recipe['name'] + ':' + view
            evidence = matches[matches.recipe.eq(recipe['name']) & matches.view.eq(view)]
            final, actions = apply_overlay(baseline, evidence, confidence, policy, identity)
            values[name] = metric_vector(final, labels, identities)
            final.to_csv(OUT / ('development-' + name.replace(':', '-') + '.csv'), index=False)
            actions.to_csv(OUT / ('development-' + name.replace(':', '-') + '-actions.csv'), index=False)
            summaries[name] = dict(comparison=paired_summary(values['P01'], values[name]),
                truth_recovered=int((values[name] > 0).sum()), actions=actions.reason.value_counts().to_dict())
    winner = select_recipe({r['name']: float(values[r['name'] + ':all_reference'].mean()) for r in recipes}, recipes)
    new_candidate = winner['name'] != 'P16'
    development = dict(completed_utc=datetime.now(timezone.utc).isoformat(), queries=40,
        sources=labels.source.value_counts().to_dict(), baseline_mrr25=float(values['P01'].mean()),
        baseline_truths=int((values['P01'] > 0).sum()), candidates=summaries, winner=winner,
        new_candidate_selected=new_candidate, controls={name: paired_summary(values['P01'], values[name]) for name in ['P14', 'P15']},
        truth_absent_mrr=0, reserved_labels_read=False, reserved_metrics_computed=False,
        development_protocol_sha256=sha(OUT / 'development-protocol.json'),
        search_report_sha256=sha(OUT / 'search-report.json'), cloud_report_sha256=sha(CLOUD / 'c10_report.json'),
        automatic_submission=False, submission_qualified=False,
        limitation='External CASMI2022 source/CE shift and historical-pretraining uncertainty. Development selection alone does not qualify a submission.')
    pd.DataFrame(dict(molecule_id=labels.index, **values)).to_csv(OUT / 'development-predictions.csv', index=False)
    write(OUT / 'development.json', development)
    freeze = dict(frozen_utc=datetime.now(timezone.utc).isoformat(), winner=winner,
        new_candidate_selected=new_candidate, development_sha256=sha(OUT / 'development.json'),
        development_protocol_sha256=sha(OUT / 'development-protocol.json'),
        config_sha256=sha(CONFIG), policy_sha256=sha(POLICY),
        source_sha256=protocol['source_sha256'], reference_manifest_sha256=sha(POOL / 'reference-manifest.json'),
        matches_sha256=sha(OUT / 'matches.parquet'), cloud_report_sha256=sha(CLOUD / 'c10_report.json'),
        reserved_labels_read=False, reserved_metrics_computed=False, automatic_submission=False,
        next_action='Evaluate frozen winner on120 reserved only if a new candidate wins. No reselection; real smoke and independent audit remain required.')
    write(OUT / 'selection-freeze.json', freeze)
    print('DONE C10 DEVELOPMENT FREEZE', json.dumps(development), flush=True)


if __name__ == '__main__': main()
