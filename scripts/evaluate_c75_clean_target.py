"""Seal both unchanged C71 predictions, then consume the entire C72 panel once.

Admission is exactly the panel's existing predeclared whole512/mode policy.
Passing this source check still requires production and once-only confirmation.
"""
import os
for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ.setdefault(name, '2')
import torch
torch.set_num_threads(2)
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from c75_port_utils import ROOT, read, write, sha, stamp
from complete_c75_local_target import BASE, OUT, PAIR, PANEL, COLLECTION, PLAN, native, verify, load_c74
from evaluate_c45_complete import point
from validation_repair import clustered_summary

FREEZE = OUT / 'clean-evaluation-freeze.json'
RESULT = OUT / 'clean-source-results.json'


def paths():
    result = {name: OUT / (name+'.csv') for name in ['P01', 'P14', 'P16']}
    result['physical_parent'] = OUT / 'physical-parent.csv'
    result.update({trial: OUT / 'clean-head-predictions' / (trial+'.csv') for trial in ['cpu', 'gpu']})
    return result


def verify_freeze():
    auth = read(ROOT / '.local/c38-active-goal.json')
    assert auth['active'] and auth['submissions_used'] == 0 and auth['scientific_gate_hold']
    frozen = read(FREEZE)
    assert frozen['source_sha256'] == sha(__file__)
    for name, pin in frozen['files_sha256'].items(): assert sha(native(name)) == pin, name
    assert not (ROOT / 'outputs/local/c13-confirmation-20261110/evaluation-consumption.json').exists()
    return frozen


def freeze():
    verify(); assert not FREEZE.exists() and not RESULT.exists()
    assert read(BASE / 'fixture-parity-audit.json')['passed']
    assert read(BASE / 'full-port/independent-run-audit.json')['passed']
    assert read(ROOT / 'outputs/local/c69-clean-parent-evidence-oof-20261112/independent-oof-audit.json')['CPU_gate_passed']
    assert read(ROOT / 'outputs/local/c70-clean-gpu-parent-evidence-oof-20261112/independent-oof-audit.json')['GPU_gate_passed']
    heads = OUT / 'clean-head-predictions'; receipt = read(heads / 'score-receipt.json')
    assert receipt['source_sha256'] == sha(ROOT / 'scripts/complete_c75_local_target.py')
    assert receipt['complete_pipeline_binding_sha256'] == sha(PAIR / 'complete-pipeline-binding.json')
    assert receipt['adapter_freeze_sha256'] == sha(PLAN) and receipt['queries'] == 512 and receipt['acquisitions'] == 2215
    assert set(receipt['trials']) == {'cpu', 'gpu'} and not receipt['new_source_quality_read']
    for trial, entry in receipt['trials'].items():
        assert entry['model_sha256'] == sha(PAIR / (trial+'-model.npz'))
        assert entry['independent_scores_and_lists_exact'] and entry['membership_preserved'] and entry['all_formula_slots_preserved']
    for name, pin in receipt['files_sha256'].items(): assert sha(heads / name) == pin
    files = [Path(__file__), ROOT / 'scripts/audit_c75_clean_quality.py',
             ROOT / 'scripts/evaluate_c45_complete.py', ROOT / 'scripts/audit_c45_quality.py',
             ROOT / 'scripts/validation_repair.py', PLAN, BASE / 'fixture-parity-audit.json',
             BASE / 'full-port/independent-run-audit.json', PAIR / 'complete-pipeline-binding.json',
             PAIR / 'complete-candidate-freeze.json', PAIR / 'independent-fitted-pair-audit.json',
             PAIR / 'cpu-model.npz', PAIR / 'gpu-model.npz',
             OUT / 'complete-parent-freeze.json', OUT / 'local-parent-adapter-receipt.json',
             OUT / 'target-parent/independent-parent-audit.json',
             OUT / 'target-masked-reference/mask-manifest.json',
             OUT / 'target-masked-reference/independent-mask-audit.json',
             PANEL / 'validation-freeze.json', PANEL / 'preparation-protocol.json',
             PANEL / 'independent-acquisition-audit.json', PANEL / 'raw-source-alias-audit.json',
             PANEL / 'fitted-query-alias-audit.json', OUT / 'queries.parquet',
             COLLECTION / 'collection-audit.json', COLLECTION / 'acquisition-fidelity-audit.json',
             COLLECTION / 'independent-core-health-audit.json', heads / 'score-receipt.json']
    files.extend(heads / name for name in receipt['files_sha256'])
    identity = load_c74().module('c75_quality_identity', 'c09e_features.py').structure
    identities = {}
    ids = pd.read_csv(paths()['P01'], dtype=str, keep_default_na=False).molecule_id.tolist()
    assert len(ids) == len(set(ids)) == 512
    for path in paths().values():
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        assert frame.molecule_id.tolist() == ids; files.append(path)
        for text in frame.smiles:
            for s in text.split(';'):
                if s not in identities: identities[s] = identity(s)[0]
    identity_path = OUT / 'clean-quality-identities.parquet'; assert not identity_path.exists()
    pd.DataFrame([dict(smiles=s, key=k) for s, k in identities.items()]).to_parquet(identity_path, index=False)
    files.append(identity_path)
    panel = read(PANEL / 'validation-freeze.json')
    assert panel['complete_pipeline_binding_sha256'] == sha(PAIR / 'complete-pipeline-binding.json')
    # Labels are only hashed at this stage. No prediction/truth comparison.
    assert sha(PANEL / 'labels.csv') == panel['files_sha256'][r'outputs\local\c72-clean-target-holdout-20261112\labels.csv']
    write(FREEZE, dict(frozen_utc=stamp(), source_sha256=sha(__file__),
        files_sha256={str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in files},
        labels_sha256=sha(PANEL / 'labels.csv'), queries=512, acquisitions=2215,
        admission_policy=read(PANEL / 'preparation-protocol.json')['quality_gate'],
        both_complete_predictions_frozen=True, source_quality_read=False, C13_read=False,
        production_and_confirmation_required_before_scoring=True))
    print('C75 WHOLE512 BOTH PREDICTIONS SEALED; QUALITY UNREAD', flush=True)


def evaluate(resume=False):
    frozen = verify_freeze()
    marker_path = PANEL / 'evaluation-consumption.json'
    marker = dict(evaluation_freeze_sha256=sha(FREEZE), labels_sha256=frozen['labels_sha256'],
                  candidates=['cpu', 'gpu'], source='Entire frozen512C72queries/2215originalacquisitions', C13_read=False)
    assert sha(PANEL / 'labels.csv') == frozen['labels_sha256']
    if marker_path.exists():
        prior = read(marker_path)
        assert resume and {k: prior[k] for k in marker} == marker, 'Only the exact committed evaluation may resume'
        if RESULT.exists():
            assert read(RESULT)['evaluation_freeze_sha256'] == sha(FREEZE)
            print('C75 EXACT COMMITTED QUALITY RESULTS ALREADY COMPLETE', flush=True); return
    else:
        assert not resume and not RESULT.exists()
        with marker_path.open('x', encoding='utf8') as stream:
            json.dump(dict(started_utc=stamp(), **marker), stream, indent=2)
    frames = {name: pd.read_csv(path, dtype=str, keep_default_na=False) for name, path in paths().items()}
    mids = frames['P01'].molecule_id.tolist(); assert len(set(mids)) == 512
    labels = pd.read_csv(PANEL / 'labels.csv', dtype=str).set_index('molecule_id').loc[mids]
    assert len(labels) == 512 and labels.source.value_counts().to_dict() == {'enveda-180': 512}
    assert labels.group.nunique() == 471
    identities = pd.read_parquet(OUT / 'clean-quality-identities.parquet').set_index('smiles').key.to_dict()
    lists = {}; ranks = {}
    for name, frame in frames.items():
        assert frame.molecule_id.tolist() == mids
        lists[name] = [[identities[s] for s in text.split(';')] for text in frame.smiles]
        assert all(1 <= len(row) <= 25 and len(set(row)) == len(row) for row in lists[name])
        ranks[name] = np.array([row.index(truth)+1 if truth in row else 0
                               for row, truth in zip(lists[name], labels.key)], dtype=np.int16)
    q = pd.read_parquet(OUT / 'queries.parquet'); assert len(q) == 2215
    modes = q.groupby('molecule_id').ionization_mode.agg(lambda v: 'mixed' if v.nunique()>1 else v.iloc[0]).loc[mids].to_numpy()
    assert pd.Series(modes).value_counts().to_dict() == {'positive': 353, 'mixed': 124, 'negative': 35}
    assert (modes == labels.query_mode.to_numpy()).all()
    groups = labels.group.to_numpy(); whole = np.ones(512, bool)
    rr = {name: np.divide(1., values, out=np.zeros(512), where=values>0) for name, values in ranks.items()}
    def summary(base, trial, mask):
        result = clustered_summary(rr[base][mask], rr[trial][mask], groups[mask], 10000, 20261175)
        result['point'] = point(ranks[base], ranks[trial], mask)
        return result
    def compare(base, trial):
        return dict(all=summary(base, trial, whole),
            actual_query_mode={mode: summary(base, trial, modes==mode) for mode in sorted(set(modes))},
            recovered_truths=int(((ranks[base]==0)&(ranks[trial]>0)).sum()),
            displaced_truths=int(((ranks[base]>0)&(ranks[trial]==0)).sum()),
            truth_absent_queries=int((ranks[trial]==0).sum()),
            rank1_changes=sum(a[0]!=b[0] for a,b in zip(lists[base], lists[trial])),
            candidate_membership_changes=sum(set(a)!=set(b) for a,b in zip(lists[base], lists[trial])))
    results = {}
    for trial in ['cpu', 'gpu']:
        assert all(set(a)==set(b) for a,b in zip(lists['physical_parent'], lists[trial]))
        comparisons = {base: compare(base, trial) for base in ['P01', 'P16', 'physical_parent']}
        passed = all(v['all']['point']['strictly_positive'] and
                     all(s['point']['nonnegative'] for s in v['actual_query_mode'].values()) for v in comparisons.values())
        results[trial] = dict(comparisons=comparisons, admission=dict(passed=passed, statistical_promotion=False))
    np.savez(OUT / 'clean-source-ranks.npz', **ranks)
    write(RESULT, dict(completed_utc=stamp(), source_sha256=sha(__file__), evaluation_freeze_sha256=sha(FREEZE),
        consumption_sha256=sha(marker_path), queries=512, acquisitions=2215, chemistry_groups=471,
        actual_query_modes=pd.Series(modes).value_counts().to_dict(), results=results,
        P16_overlay=compare('P01', 'P16'), physical_adapter=compare('P16', 'physical_parent'),
        P14_negative_control=compare('P01', 'P14'),
        passed_pair=all(v['admission']['passed'] for v in results.values()),
        absent_truths_zero=True, all_queries_retained=True, C13_read=False, scoring_entries=0,
        qualified_for_scoring=False,
        limitation='Single-source controlled in-corpus check; historic public parent/encoder exposure persists. No public-score calibration, cohort rescue, model refit or repeated confirmation is permitted from this outcome.'))
    print('C75 WHOLE512 SOURCE ADMISSION', {t: v['admission'] for t,v in results.items()}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=['freeze', 'evaluate'])
    parser.add_argument('--resume-same-evaluation', action='store_true'); args = parser.parse_args()
    if args.phase == 'freeze': freeze()
    else: evaluate(args.resume_same_evaluation)
