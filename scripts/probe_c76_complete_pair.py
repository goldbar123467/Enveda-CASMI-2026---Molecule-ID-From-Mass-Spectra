"""Run the deployable complete head path on USED engineering inputs only."""
import os
for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[name] = '2'
import torch
import argparse
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
from c75_port_utils import ROOT, read, write, sha, stamp

BASE = ROOT / 'outputs/local/c76-numerical-portability-20261112'
OUT = BASE / 'complete-pair-used-probe'
BUNDLE = ROOT / 'outputs/publish/c76-complete-direct-pair-20261112'
WORK = Path('/home/imsa/.local/share/enveda-c76/complete-pair-used/working')
SOURCE = Path('/home/imsa/.local/share/enveda-c76/fixture-resource/working')
REFERENCE = ROOT / 'outputs/local/c74-complete-clean-target-20261112/target-masked-reference'
CODE = ROOT / 'outputs/local/c45-complete-parent-validation-20261112/frozen-p16-code'


def main():
    assert not OUT.exists() and not WORK.exists()
    assert 'enveda-c76-complete-used' in Path('/proc/self/cgroup').read_text()
    available = next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
    assert available >= 8 * 2**30, 'Eight GiB start reserve required'
    audit = read(BASE / 'fixture-resource-port/independent-run-audit.json')
    assert audit['C76_engineering_fixture_passed'] and not audit['statistical_qualification']
    assert read(BASE / 'production-preparation/independent-source-audit.json')['passed']
    assert not (ROOT / 'outputs/local/c72-clean-target-holdout-20261112/evaluation-consumption.json').exists()
    assert not (ROOT / 'outputs/local/c13-confirmation-20261110/evaluation-consumption.json').exists()
    manifest = read(BUNDLE / 'pair-manifest.json')
    for name, pin in manifest['files_sha256'].items(): assert sha(BUNDLE / name) == pin, name
    spec = importlib.util.spec_from_file_location('c76_original_c74_audit', ROOT / 'scripts/c74_complete_target.py')
    c74 = importlib.util.module_from_spec(spec); spec.loader.exec_module(c74)
    contract = c74.frozen_contract()
    mask = read(REFERENCE / 'mask-manifest.json')
    for name, pin in {**mask['reference_files'], **mask['audit_files']}.items(): assert sha(REFERENCE / name) == pin, name
    copies = {'submission.csv': 'c41_p01.csv', 'effective_query_spectra.parquet': 'c41_queries.parquet',
        'neural_candidate_features.parquet': 'c41_neural_p01.parquet',
        'neural_public.parquet': 'neural_public.parquet', 'neural_ours.parquet': 'neural_ours.parquet',
        'frozen_candidate_evidence.parquet': 'frozen_candidate_evidence.parquet',
        'outsider_neural_features.parquet': 'outsider_neural_features.parquet',
        'execution_receipt.json': 'execution_receipt.json'}
    inputs = {str(SOURCE / source): sha(SOURCE / source) for source in copies.values()}
    OUT.mkdir(); WORK.mkdir(parents=True)
    write(OUT / 'intent.json', dict(created_utc=stamp(), source_sha256=sha(__file__),
        queries=12, acquisitions=51, engineering_USED_inputs_only=True,
        pair_manifest_sha256=sha(BUNDLE / 'pair-manifest.json'), inputs_sha256=inputs,
        P16_mask_manifest_sha256=sha(REFERENCE / 'mask-manifest.json'),
        no_quality_or_truth_read=True, C72_prediction=False, C13_read=False,
        actual_local_CPU=True, actual_T4_smoke=False))
    for target, source in copies.items():
        shutil.copyfile(SOURCE / source, WORK / target)
        assert sha(WORK / target) == inputs[str(SOURCE / source)]
    for name in list(contract['source_files']) + ['c09e_policy.json']:
        shutil.copyfile(CODE / name, WORK / name)
    contract.update(reference_files=mask['reference_files'], validation_only=True,
        reference_mask_manifest_sha256=sha(REFERENCE / 'mask-manifest.json'),
        original_deployment_sha256=sha(CODE / 'deployment.json'))
    write(WORK / 'validation-deployment.json', contract)
    with (OUT / 'P16.log').open('w') as log:
        subprocess.run([sys.executable, str(WORK / 'apply_external_reference.py'), '--work', str(WORK),
            '--input-root', str(REFERENCE), '--contract', str(WORK / 'validation-deployment.json')],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    p16_audit = c74.audit_replay(WORK)
    assert p16_audit['queries'] == 12
    shutil.copyfile(WORK / 'submission.csv', WORK / 'c76_p16.csv')
    with (OUT / 'pair.log').open('w') as log:
        subprocess.run([sys.executable, str(BUNDLE / 'score_c76_complete_pair.py'), '--work', str(WORK),
            '--bundle', str(BUNDLE), '--reference', str(REFERENCE), '--output', str(WORK / 'c76_scores')],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    receipt = read(WORK / 'c76_scores/score-receipt.json')
    assert receipt['queries'] == 12 and receipt['acquisitions'] == 51
    assert set(receipt['trials']) == {'cpu', 'gpu'}
    for item in receipt['trials'].values():
        assert item['independent_scores_and_lists_exact'] and item['changed_lists'] > 0
        assert item['candidate_membership_changes'] == item['formula_position_changes'] == item['official_identity_duplicates'] == 0
    for path, pin in inputs.items(): assert sha(path) == pin
    assert sha(__file__) == read(OUT / 'intent.json')['source_sha256']
    shutil.copyfile(WORK / 'c76_scores/score-receipt.json', OUT / 'score-receipt.json')
    write(OUT / 'results.json', dict(completed_utc=stamp(), source_sha256=sha(__file__), passed=True,
        intent_sha256=sha(OUT / 'intent.json'), pair_manifest_sha256=sha(BUNDLE / 'pair-manifest.json'),
        queries=12, acquisitions=51, real_original_P16_independent_overlay_exact=True,
        shared_production_and_validation_head_entry_point_executed=True,
        both_heads_changed_real_same_formula_rankings=True,
        independent_score_and_formula_slot_reconstruction=True,
        score_receipt_sha256=sha(OUT / 'score-receipt.json'),
        actual_local_CPU=True, actual_T4_smoke=False, engineering_inputs_only=True,
        new_source_quality_read=False, C13_read=False, scoring_entries=0,
        qualified_for_scoring=False, working=str(WORK)))
    print('C76 COMPLETE P16/PHYSICAL/CPU/GPU USED PATH PASSED', flush=True)


if __name__ == '__main__': main()
