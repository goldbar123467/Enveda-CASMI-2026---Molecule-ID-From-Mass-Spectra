"""Frozen complete C76/control replay and joint prediction seal; no truth input."""
import argparse
import importlib.util
import shutil
import subprocess
import sys
from c75_port_utils import ROOT, read, write, sha, stamp

BASE = ROOT / 'outputs/local/c76-numerical-portability-20261112'
BINDING = BASE / 'complete-pipeline-binding-r1.json'
PANEL = ROOT / 'outputs/local/c72-clean-target-holdout-20261112'
OLD_MASK = ROOT / 'outputs/local/c74-complete-clean-target-20261112/target-masked-reference'
BUNDLE = ROOT / 'outputs/publish/c76-complete-direct-pair-r1-20261112'
CLOUDS = {arm: ROOT / 'outputs/forks' / (slot + '-v1') for arm, slot in [
    ('legacy', 'c73-full-masked-validation'), ('c76', 'c76-full-masked-validation')]}


def portable_read(path):
    value = read(path)
    if isinstance(value, dict) and 'files_sha256' in value:
        old = value['files_sha256']
        value['files_sha256'] = {name.replace('\\', '/'): pin for name, pin in old.items()}
        assert len(old) == len(value['files_sha256'])
    return value


def verify():
    auth = read(ROOT / '.local/c38-active-goal.json')
    assert auth['active'] and auth['submissions_used'] == 0
    assert not (PANEL / 'evaluation-consumption.json').exists()
    assert not (ROOT / 'outputs/local/c13-confirmation-20261110/evaluation-consumption.json').exists()
    binding = read(BINDING)
    for name, pin in binding['files_sha256'].items(): assert sha(ROOT / name) == pin, name
    for arm, cloud in CLOUDS.items():
        collection = read(cloud / 'collection-audit.json')
        assert collection['actual_two_T4s'] and collection['remote_source_exact']
        assert collection['queries'] == 512 and collection['forbidden_reference_rows'] == 0
        assert collection['baseline_replay_exact'] and not collection['quality_metrics_read']
        for name, pin in collection['files_sha256'].items(): assert sha(cloud / name) == pin, name
        fidelity = read(cloud / 'acquisition-fidelity-audit.json')
        assert fidelity['queries'] == 512 and fidelity['acquisitions'] == 2215
        assert fidelity['all_original_acquisitions_retained'] and fidelity['all_original_metadata_and_peaks_exact']
        assert read(cloud / 'independent-core-health-audit.json')['passed']
    return binding


def load_c74(arm):
    spec = importlib.util.spec_from_file_location('c76_unchanged_c74_' + arm, ROOT / 'scripts/c74_complete_target.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.OUT = BASE / 'source-parents' / arm
    module.CLOUD = CLOUDS[arm]; module.EXTRA = CLOUDS[arm]
    module.read = portable_read
    return module


def replay(arm):
    verify()
    module = load_c74(arm); out = module.OUT
    assert not out.exists(), 'Inspect any partial parent instead of overwriting'
    out.mkdir(parents=True)
    shutil.copytree(OLD_MASK, out / 'target-masked-reference')
    shutil.copyfile(BINDING, out / 'candidate-family-freeze.json')
    module.replay_c39()
    write(out / 'C76-provenance.json', dict(completed_utc=stamp(), source_sha256=sha(__file__),
        complete_C76_binding_sha256=sha(BINDING), control_arm=arm,
        original_C74_functions_and_policy_unchanged=True,
        P01_source=('Original unrepaired R3 on actual T4' if arm == 'legacy' else 'Explicit C76 numerical recipe on actual T4'),
        adapted_globals=['OUT', 'CLOUD', 'EXTRA', 'read:files_sha256 path keys only'],
        collection_audit_sha256=sha(CLOUDS[arm] / 'collection-audit.json'),
        complete_parent_freeze_sha256=sha(out / 'complete-parent-freeze.json'),
        quality_read=False, C13_read=False))
    print('C76 CONTROLLED COMPLETE PARENT REPLAYED', arm, flush=True)


def score():
    verify()
    for arm in CLOUDS:
        out = BASE / 'source-parents' / arm
        assert read(out / 'C76-provenance.json')['complete_C76_binding_sha256'] == sha(BINDING)
        for name, pin in read(out / 'complete-parent-freeze.json')['files_sha256'].items(): assert sha(out / name) == pin, name
    out = BASE / 'source-parents/c76'; work = out / 'target-parent'
    for target, source in {'neural_public.parquet': 'neural_public.parquet',
        'neural_ours.parquet': 'neural_ours.parquet', 'execution_receipt.json': 'execution_receipt.json'}.items():
        assert not (work / target).exists(); shutil.copyfile(CLOUDS['c76'] / source, work / target)
    shutil.copyfile(work / 'submission.csv', work / 'c76_p16.csv')
    subprocess.run([sys.executable, str(BUNDLE / 'score_c76_complete_pair.py'), '--work', str(work),
        '--bundle', str(BUNDLE), '--reference', str(out / 'target-masked-reference'),
        '--output', str(BASE / 'source-pair-predictions')], check=True, cwd=ROOT)
    import torch
    import pandas as pd
    pd.testing.assert_frame_equal(pd.read_csv(out / 'physical-parent.csv', dtype=str),
        pd.read_csv(BASE / 'source-pair-predictions/physical-parent.csv', dtype=str), check_exact=True)
    print('C76 BOTH COMPLETE SOURCE PREDICTION SETS WRITTEN; QUALITY SEALED', flush=True)


def seal():
    verify()
    target = BASE / 'joint-source-prediction-freeze.json'; assert not target.exists()
    receipt = read(BASE / 'source-pair-predictions/score-receipt.json')
    assert receipt['queries'] == receipt['valid_rows'] == 512 and receipt['acquisitions'] == 2215
    assert set(receipt['trials']) == {'cpu', 'gpu'} and not receipt['quality_labels_read']
    files = []
    controls = {}
    for arm in CLOUDS:
        out = BASE / 'source-parents' / arm
        controls[arm] = {}
        for name in ['P01.csv', 'P16.csv', 'physical-parent.csv']:
            path = out / name; files.append(path)
            controls[arm][name] = path.relative_to(ROOT).as_posix()
        files += [out / 'complete-parent-freeze.json', out / 'C76-provenance.json',
            CLOUDS[arm] / 'collection-audit.json', CLOUDS[arm] / 'acquisition-fidelity-audit.json',
            CLOUDS[arm] / 'independent-core-health-audit.json']
    for name, pin in receipt['files_sha256'].items():
        path = BASE / 'source-pair-predictions' / name; assert sha(path) == pin; files.append(path)
    files.append(BASE / 'source-pair-predictions/score-receipt.json')
    write(target, dict(frozen_utc=stamp(), source_sha256=sha(__file__), complete_pipeline_binding_sha256=sha(BINDING),
        controls=controls, candidates={trial: (BASE / 'source-pair-predictions' / (trial + '.csv')).relative_to(ROOT).as_posix() for trial in ['cpu', 'gpu']},
        files_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in files},
        queries=512, acquisitions=2215, all_six_controls_and_both_heads_frozen_before_quality=True,
        quality_protocol_sha256=sha(PANEL / 'preparation-protocol.json'),
        controls_addendum_sha256=sha(ROOT / 'configs/c76-source-controls-addendum-v1.json'),
        no_source_quality_read=True, C13_read=False, scoring_entries=0, qualified_for_scoring=False))
    print('C76 JOINT SOURCE PREDICTIONS FROZEN; SINGLE JOINT QUALITY READ STILL REQUIRED', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=['replay', 'score', 'seal'])
    parser.add_argument('--arm', choices=['legacy', 'c76']); args = parser.parse_args()
    if args.phase == 'replay': assert args.arm; replay(args.arm)
    elif args.phase == 'score': score()
    else: seal()
