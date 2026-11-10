"""Prospective reserved120 check of the frozen C12 recipe; never submits."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2'); os.environ.setdefault('MKL_NUM_THREADS', '2')
import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow as pa
from rdkit import rdBase

from precision_cache import ROOT, sha, write, features
from prepare_c12_pool import OUT, CONFIG, CLOUD
from prefix_expansion import expand
from finish_c09_confirmation import metric_vector
from tail_fusion import paired_summary
from confirm_c10_candidate import statistical_gate

LOCAL = ROOT / 'outputs/local/c10-casmi2022-20261110'


def protocol():
    assert not (OUT / 'confirmation-protocol.json').exists()
    assert not (OUT / 'confirmation.json').exists()
    freeze = json.loads((OUT / 'selection-freeze.json').read_text())
    assert freeze['new_candidate_selected'] and not freeze['reserved_prediction_metrics_computed']
    names = ['confirm_c12_pool.py', 'develop_c12_pool.py', 'prepare_c12_pool.py', 'score_c12_pool.py',
        'prefix_expansion.py', 'confirm_c10_candidate.py', 'tail_fusion.py', 'finish_c09_confirmation.py']
    write(OUT / 'confirmation-protocol.json', dict(prepared_utc=datetime.now(timezone.utc).isoformat(),
        selection_freeze_sha256=sha(OUT / 'selection-freeze.json'), config_sha256=sha(CONFIG),
        source_contract_sha256=sha(OUT / 'source-contract.json'), source_sha256={n: sha(ROOT / 'scripts' / n) for n in names},
        labels_sha256=sha(LOCAL / 'labels.csv'), partitions_sha256=sha(LOCAL / 'partitions.json'),
        required_queries=120, selection='One frozen recipe; no reserved reselection or coefficient search',
        gate='Positive paired95% lower bound against both completeP01 andP16; nonnegative alias sensitivity; retain all missing-truth zeros',
        bootstrap_draws=10000, bootstrap_seed=20261105,
        negative_controls='Report P14/P15 even if they contradict actual public regressions',
        reserved_prediction_metrics_read=False, automatic_submission=False,
        full_submission_qualification='Real Kaggle changed-path smoke, independent execution/parity/source/identity/fallback audit and actual P16 resolution still required'))
    print('DONE C12 CONFIRMATION PROTOCOL', sha(OUT / 'confirmation-protocol.json'), flush=True)


def evaluate():
    pa.set_cpu_count(2); assert rdBase.rdkitVersion == '2026.03.3'
    assert not (OUT / 'confirmation.json').exists() and not (OUT / 'confirmation-input-freeze.json').exists()
    spec = json.loads((OUT / 'confirmation-protocol.json').read_text())
    for name, pin in spec['source_sha256'].items(): assert sha(ROOT / 'scripts' / name) == pin, name
    for path, key in [(CONFIG, 'config_sha256'), (OUT / 'source-contract.json', 'source_contract_sha256'),
        (OUT / 'selection-freeze.json', 'selection_freeze_sha256'), (LOCAL / 'labels.csv', 'labels_sha256'),
        (LOCAL / 'partitions.json', 'partitions_sha256')]: assert sha(path) == spec[key], key
    freeze = json.loads((OUT / 'selection-freeze.json').read_text())
    assert freeze['new_candidate_selected'] and freeze['development_report_sha256'] == sha(OUT / 'development.json')
    full = OUT / 'all'; receipt = json.loads((full / 'scoring-receipt.json').read_text())
    assert receipt['scope'] == 'all' and receipt['queries'] == 160 and receipt['input_errors'] == 0
    assert receipt['source_contract_sha256'] == spec['source_contract_sha256']
    assert receipt['scores_sha256'] == sha(full / 'scores.parquet')
    assert not receipt['prediction_metrics_computed'] and not receipt['query_labels_read']
    config = json.loads(CONFIG.read_text())
    assert all(x['max_raw_score_abs_difference'] <= config['local_raw_score_parity_tolerance'] for x in receipt['local_cloud_parity'].values())
    parent = pd.read_csv(full / 'parent.csv', dtype=str, keep_default_na=False)
    scores = pd.read_parquet(full / 'scores.parquet')
    final, actions = expand(parent, scores, config)
    partitions = json.loads((LOCAL / 'partitions.json').read_text())
    development = final.set_index('molecule_id').loc[partitions['development']].reset_index()
    expected = pd.read_csv(OUT / 'development.csv', dtype=str, keep_default_na=False)
    assert development.equals(expected), 'Full-scope implementation changed development output; do not read reserved metrics'
    # Parent reference-input exclusion removes no rows on this panel. Verify the
    # two saved match views exactly, rather than choosing a sensitivity on labels.
    match_path = ROOT / 'outputs/local/c10-candidates-20261110/matches.parquet'
    matches = pd.read_parquet(match_path); matches = matches[matches.recipe.eq('P16')]
    columns = ['molecule_id', 'cosine', 'matched_peaks', 'smiles', 'reference_row']
    pd.testing.assert_frame_equal(matches[matches.view.eq('all_reference')][columns].reset_index(drop=True),
        matches[matches.view.eq('input_alias_excluded')][columns].reset_index(drop=True), check_exact=True)
    search = json.loads((ROOT / 'outputs/local/c10-candidates-20261110/search-report.json').read_text())
    assert search['query_input_alias_reference_rows'] == 0
    inputs = [full / 'scores.parquet', full / 'parent.csv', full / 'scoring-receipt.json',
        CLOUD / 'c10_p01.csv', CLOUD / 'c10_p14.csv', ROOT / 'outputs/local/c10-candidates-20261110/p15-control.csv', match_path]
    write(OUT / 'confirmation-input-freeze.json', dict(frozen_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=sha(OUT / 'confirmation-protocol.json'), exact_development_output_parity=True,
        source_alias_views_identical=True, reserved_prediction_metrics_read=False,
        input_sha256={str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in inputs}))
    labels = pd.read_csv(LOCAL / 'labels.csv', dtype=str, skiprows=range(1, 41)).set_index('molecule_id')
    assert len(labels) == 120 and labels.partition.eq('confirmation').all()
    assert labels.index.tolist() == partitions['confirmation']
    identities = dict(zip(scores.smiles.astype(str), scores.metric_key.astype(str)))
    def identity(smiles):
        if smiles not in identities: identities[smiles] = features.structure(smiles)[0]
        return identities[smiles]
    def subset(frame): return frame.set_index('molecule_id').loc[labels.index].reset_index()
    frames = dict(P16=subset(parent), C12=subset(final))
    for name, path in [('P01', CLOUD / 'c10_p01.csv'), ('P14', CLOUD / 'c10_p14.csv'),
        ('P15', ROOT / 'outputs/local/c10-candidates-20261110/p15-control.csv')]:
        frames[name] = subset(pd.read_csv(path, dtype=str, keep_default_na=False))
    for frame in frames.values():
        for row in frame.itertuples(index=False):
            for smiles in row.smiles.split(';'): identity(smiles)
    values = {name: metric_vector(frame, labels, identities) for name, frame in frames.items()}
    comparisons = {'winner_vs_' + name: paired_summary(values[name], values['C12']) for name in ['P01', 'P16']}
    comparisons.update({'alias_excluded_' + name: value for name, value in list(comparisons.items())})
    gate = statistical_gate(comparisons)
    controls = {name: paired_summary(values['P01'], values[name]) for name in ['P14', 'P15']}
    report = dict(completed_utc=datetime.now(timezone.utc).isoformat(), queries=120, candidate='P16_plus_deep_prefix',
        comparisons=comparisons, statistical_gate=gate, controls=controls,
        detects_known_public_regression={name: value['delta'] < 0 for name, value in controls.items()},
        negative_control_interpretation='Control signs are reported without query filtering. Failure to reproduce a public regression limits hidden-score calibration even if incremental support passes.',
        truth_coverage={name: int((value > 0).sum()) for name, value in values.items()},
        absent_truths_recovered=int(((values['P16'] == 0) & (values['C12'] > 0)).sum()),
        truths_lost=int(((values['P16'] > 0) & (values['C12'] == 0)).sum()),
        truth_absent_mrr=0, reserved_recipe_reselection=False, source_alias_views_identical=True,
        full_scope_development_output_exact=True,
        selection_freeze_sha256=sha(OUT / 'selection-freeze.json'), protocol_sha256=sha(OUT / 'confirmation-protocol.json'),
        input_freeze_sha256=sha(OUT / 'confirmation-input-freeze.json'),
        automatic_submission=False, submission_qualified=False,
        next_action='Proceed toward real Kaggle smoke/independent review only if the statistical gate passes. No public trial while P16 remains pending. Otherwise reject this frozen recipe.',
        limitation='External positive-only CASMI2022/unknownCE shift, incomplete parent reference provenance and possible historical foundation exposure; statistical support is not a hidden-score guarantee.')
    frames['C12'].to_csv(OUT / 'confirmation.csv', index=False)
    actions[actions.molecule_id.isin(labels.index)].to_csv(OUT / 'confirmation-actions.csv', index=False)
    pd.DataFrame(dict(molecule_id=labels.index, **values)).to_csv(OUT / 'confirmation-predictions.csv', index=False)
    write(OUT / 'confirmation.json', report)
    print('DONE C12 FROZEN CONFIRMATION', json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare-protocol', action='store_true'); group.add_argument('--evaluate', action='store_true')
    args = parser.parse_args(); protocol() if args.prepare_protocol else evaluate()
