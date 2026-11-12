"""Replay complete controls/heads, freeze evidence, then seal both predictions."""
import argparse
import copy
import importlib.util
import shutil
import subprocess
import sys
import types
from c75_port_utils import ROOT, read, write, sha, stamp
from c76_submission_contract import BINDING, CONFIRM, FAMILY, verify_files
from c76_confirmation_preparation import ARMS, ORIGINAL, PREPARATION
from c76_confirmation_inference import BUNDLE, CLOUDS, PLAN, collected, verify_plan

PARENTS = CONFIRM / 'parents'
PREDICTIONS = CONFIRM / 'predictions'
PRE_FREEZE = CONFIRM / 'pre-evaluation-artifact-freeze.json'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def replay(arm):
    cloud, collection = collected(arm)
    import torch
    import numpy as np
    import pandas as pd
    from c74_complete_target import CODE, frozen_contract, audit_replay
    mask = PREPARATION / 'masked-P16-reference'; manifest = read(mask / 'mask-manifest.json')
    reference_audit = read(PREPARATION / 'independent-reference-audit.json')
    assert reference_audit['passed'] and reference_audit['forbidden_reference_rows'] == 0
    verify_files(reference_audit['artifact_sha256'])
    for name, pin in {**manifest['reference_files'], **manifest['audit_files']}.items(): assert sha(mask / name) == pin
    out = PARENTS / arm; assert not out.exists(), 'Inspect a partial replay rather than overwrite'
    work = out / 'work'; work.mkdir(parents=True)
    shutil.copyfile(cloud / 'c41_p01.csv', work / 'submission.csv')
    for target, source in [('effective_query_spectra.parquet', 'c41_queries.parquet'),
        ('neural_candidate_features.parquet', 'c41_neural_p01.parquet'), ('neural_public.parquet', 'neural_public.parquet'),
        ('neural_ours.parquet', 'neural_ours.parquet'), ('execution_receipt.json', 'execution_receipt.json'),
        ('frozen_candidate_evidence.parquet', 'frozen_candidate_evidence.parquet'), ('outsider_neural_features.parquet', 'outsider_neural_features.parquet')]:
        shutil.copyfile(cloud / source, work / target)
    contract = frozen_contract()
    for name in list(contract['source_files']) + ['c09e_policy.json']: shutil.copyfile(CODE / name, work / name)
    contract = copy.deepcopy(contract); contract.update(reference_files=manifest['reference_files'], validation_only=True,
        reference_mask_manifest_sha256=sha(mask / 'mask-manifest.json'), original_deployment_sha256=sha(CODE / 'deployment.json'))
    write(work / 'validation-deployment.json', contract)
    with (work / 'P16-replay.log').open('w', encoding='utf8') as stream:
        subprocess.run([sys.executable, str(work / 'apply_external_reference.py'), '--work', str(work),
            '--input-root', str(mask), '--contract', str(work / 'validation-deployment.json')],
            cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    audit_replay(work)
    receipt = read(work / 'c09e_external_receipt.json')
    assert receipt['search']['queries'] == receipt['search']['spectra'] == 120
    assert receipt['search']['query_input_alias_reference_rows'] == 0
    for name, path in [('P01.csv', cloud / 'c41_p01.csv'), ('P14.csv', cloud / 'c41_p14.csv'),
        ('P16.csv', work / 'submission.csv'), ('queries.parquet', cloud / 'c41_queries.parquet')]: shutil.copyfile(path, out / name)
    shutil.copyfile(out / 'P16.csv', work / 'c76_p16.csv')
    ranker = load('c76_reserved_physical_' + arm, BUNDLE / 'c67_ranker.py')
    compatibility = load('c76_reserved_compat_' + arm, BUNDLE / 'legacy_formate_compat.py')
    chemistry = types.ModuleType('c76_reserved_neutral_' + arm)
    source = BUNDLE / 'legacy_chem_source.py'
    exec(compile(compatibility.patch_chem_source(source.read_text(encoding='utf8')), str(source), 'exec'), chemistry.__dict__)
    chosen, events = ranker.physical_parent(pd.read_csv(out / 'P01.csv', dtype=str), pd.read_csv(out / 'P16.csv', dtype=str),
        pd.read_parquet(out / 'queries.parquet'), pd.read_parquet(work / 'c09e_matches.parquet'),
        pd.read_parquet(mask / 'ext_meta.parquet'), chemistry.neutral_mass)
    assert len(chosen) == 120
    chosen.to_csv(out / 'physical-parent.csv', index=False); write(out / 'physical-parent-events.json', events)
    files = [out / name for name in ['P01.csv', 'P14.csv', 'P16.csv', 'queries.parquet', 'physical-parent.csv', 'physical-parent-events.json']]
    files += [work / name for name in ['validation-deployment.json', 'c09e_external_receipt.json', 'c09e_matches.parquet', 'independent-parent-audit.json']]
    write(out / 'parent-freeze.json', dict(frozen_utc=stamp(), source_sha256=sha(__file__), family=FAMILY, arm=arm,
        inference_plan_sha256=sha(PLAN), collection_audit_sha256=sha(cloud / 'collection-audit.json'),
        queries=120, acquisitions=120, files_sha256={path.relative_to(ROOT).as_posix(): sha(path) for path in files},
        query_labels_read=False, scoring_entries=0))
    print('C76 COMPLETE RESERVED P16/PHYSICAL PARENT REPLAYED', arm, flush=True)


def score():
    verify_plan()
    for arm in ARMS:
        collected(arm); pin = read(PARENTS / arm / 'parent-freeze.json')
        assert pin['source_sha256'] == sha(__file__) and pin['inference_plan_sha256'] == sha(PLAN)
        verify_files(pin['files_sha256'])
    assert not PREDICTIONS.exists()
    with (CONFIRM / 'heads.log').open('w', encoding='utf8') as stream:
        subprocess.run([sys.executable, str(BUNDLE / 'score_c76_complete_pair.py'), '--work', str(PARENTS / 'c76/work'),
            '--bundle', str(BUNDLE), '--reference', str(PREPARATION / 'masked-P16-reference'), '--output', str(PREDICTIONS)],
            cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    import torch
    import pandas as pd
    pd.testing.assert_frame_equal(pd.read_csv(PARENTS / 'c76/physical-parent.csv', dtype=str),
        pd.read_csv(PREDICTIONS / 'physical-parent.csv', dtype=str), check_exact=True)
    print('C76 BOTH UNCHANGED RESERVED HEADS SCORED WITHOUT LABELS', flush=True)


def freeze():
    verify_plan(); assert not PRE_FREEZE.exists()
    files = []; controls = {}
    for arm in ARMS:
        cloud, _ = collected(arm); parent = PARENTS / arm
        pin = read(parent / 'parent-freeze.json'); verify_files(pin['files_sha256'])
        assert pin['source_sha256'] == sha(__file__)
        controls[arm] = {name: (parent / name).relative_to(ROOT).as_posix() for name in ['P01.csv', 'P16.csv', 'physical-parent.csv']}
        files += [ROOT / name for name in pin['files_sha256']]
        files += [parent / 'parent-freeze.json', cloud / 'collection-audit.json']
        files += [cloud / name for name in read(cloud / 'collection-audit.json')['files_sha256']]
    receipt = read(PREDICTIONS / 'score-receipt.json')
    assert receipt['queries'] == receipt['acquisitions'] == receipt['valid_rows'] == 120
    assert set(receipt['trials']) == {'cpu', 'gpu'} and not receipt['quality_labels_read']
    for name, pin in receipt['files_sha256'].items(): assert sha(PREDICTIONS / name) == pin; files.append(PREDICTIONS / name)
    files += [PREDICTIONS / 'score-receipt.json', PLAN, PREPARATION / 'independent-reference-audit.json', PREPARATION / 'independent-notebook-audit.json']
    files = list(dict.fromkeys(files))
    write(PRE_FREEZE, dict(frozen_utc=stamp(), source_sha256=sha(__file__), family=FAMILY,
        complete_pipeline_binding_sha256=sha(BINDING), inference_plan_sha256=sha(PLAN),
        controls=controls, candidates={trial: (PREDICTIONS / (trial + '.csv')).relative_to(ROOT).as_posix() for trial in ['cpu', 'gpu']},
        files_sha256={path.relative_to(ROOT).as_posix(): sha(path) for path in files}, queries=120, acquisitions=120,
        all_six_controls_and_both_heads_frozen_before_quality=True, quality_labels_read=False))
    print('C76 ALL RESERVED CONTROL/HEAD ARTIFACTS FROZEN; INDEPENDENT EXECUTION AUDIT REQUIRED', flush=True)


def seal():
    verify_plan(); target = CONFIRM / 'joint-prediction-freeze.json'; assert not target.exists()
    frozen = read(PRE_FREEZE); verify_files(frozen['files_sha256'])
    audit_path = CONFIRM / 'execution-qualification.json'; audit = read(audit_path)
    assert audit['passed'] and audit['family'] == FAMILY and audit['inference_plan_sha256'] == sha(PLAN)
    assert audit['source_sha256'] == sha(ROOT / 'scripts/audit_c76_confirmation_execution.py')
    assert audit['pre_evaluation_artifact_freeze_sha256'] == sha(PRE_FREEZE)
    verify_files(audit['artifact_sha256'])
    write(target, dict(frozen_utc=stamp(), source_sha256=sha(__file__), family=FAMILY,
        complete_pipeline_binding_sha256=sha(BINDING), inference_plan_sha256=sha(PLAN),
        execution_qualification_sha256=sha(audit_path), original_query_sha256=sha(ORIGINAL / 'validation-queries.parquet'),
        controls=frozen['controls'], candidates=frozen['candidates'], queries=120, acquisitions=120,
        all_six_controls_and_both_heads_frozen_before_quality=True, quality_labels_read=False,
        files_sha256={**frozen['files_sha256'], PRE_FREEZE.relative_to(ROOT).as_posix(): sha(PRE_FREEZE),
            audit_path.relative_to(ROOT).as_posix(): sha(audit_path)}, scoring_entries=0))
    print('C76 BOTH RESERVED PREDICTIONS JOINTLY SEALED; ONE LABEL EVALUATION STILL REQUIRED', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=['replay', 'score', 'freeze', 'seal'])
    parser.add_argument('--arm', choices=ARMS); args = parser.parse_args()
    if args.phase == 'replay': assert args.arm; replay(args.arm)
    else: assert args.arm is None; {'score': score, 'freeze': freeze, 'seal': seal}[args.phase]()
