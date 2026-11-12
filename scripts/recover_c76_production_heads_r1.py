"""Recover only the head interface; reuse the independently proven core/P16."""
import os
for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']: os.environ[name] = '2'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import torch
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from c75_port_utils import ROOT as REPO, read, write, sha, stamp, unwrap, replace_value

BASE = REPO / 'outputs/local/c76-numerical-portability-20261112'
OUT = BASE / 'interface-repair-r1'
BUNDLE = REPO / 'outputs/publish/c76-complete-direct-pair-r1-20261112'


def main():
    assert 'enveda-c76-head-r1' in Path('/proc/self/cgroup').read_text()
    assert not (OUT / 'head-recovery-results.json').exists() and not (OUT / 'head-recovery-intent.json').exists()
    prefix = read(OUT / 'completed-prefix-audit.json'); assert prefix['completed_prefix_passed']
    working = Path(prefix['working'])
    for name, digest in prefix['reusable_unchanged_files_sha256'].items(): assert sha(working / name) == digest, name
    prep = read(OUT / 'candidate-preparation.json')
    assert sha(BUNDLE / 'pair-manifest.json') == prep['pair_manifest_sha256']
    for name, digest in read(BUNDLE / 'pair-manifest.json')['files_sha256'].items(): assert sha(BUNDLE / name) == digest, name
    assert not (working / 'submission.csv').exists()
    for name in ['c72-clean-target-holdout-20261112', 'c13-confirmation-20261110']:
        assert not (REPO / 'outputs/local' / name / 'evaluation-consumption.json').exists()
    reference = REPO / 'outputs/research/2026-11-09-silver/extlib'
    head_input = Path('/home/imsa/.local/share/enveda-c76/production-head-r1-retry/input')
    assert not head_input.exists(); head_input.mkdir(parents=True)
    for dirname, source in [('pair', BUNDLE), ('reference', reference)]:
        target = head_input / dirname; target.mkdir()
        for item in source.iterdir():
            if item.is_file(): (target / item.name).symlink_to(item.resolve())
    assert not list(head_input.rglob('validation-contract.json'))
    assert len(list(head_input.rglob('pair-manifest.json'))) == len(list(head_input.rglob('ext_meta.parquet'))) == 1
    local_pin = read(BASE / 'local-production-preview/preview-freeze.json')
    stages = {}
    for trial in ['cpu', 'gpu']:
        notebook = read(REPO / 'kaggle/forks' / ('c76-' + trial + '-production/notebook.ipynb'))
        old = read(OUT / 'previous-sources/kaggle/forks' / ('c76-' + trial + '-production/notebook.ipynb'))
        assert notebook['cells'][:28] == old['cells'][:28]
        stage = unwrap(''.join(notebook['cells'][28]['source']))
        stage = replace_value(stage, 'C76_PAYLOAD_PIN', local_pin['local_payload_sha256'])
        needle = "assert len(_devices)==2 and all('T4' in d['name'] for d in _devices)"
        assert stage.count(needle) == 1
        stage = stage.replace(needle, "assert len(_devices)==1 and '2070' in _devices[0]['name']")
        directory = 'c76_' + trial + '_scores_r1'
        stage = stage.replace('c76_scores', directory).replace('c76_pair_inference.log', 'c76_' + trial + '_pair_inference_r1.log').replace('c76_execution.json', 'c76_' + trial + '_execution_r1.json')
        compile(stage, 'c76_r1_recovered_' + trial, 'exec')
        path = OUT / (trial + '-head-stage.py'); path.write_text(stage, encoding='utf8'); stages[trial] = path
    write(OUT / 'head-recovery-intent.json', dict(started_utc=stamp(), source_sha256=sha(__file__),
        completed_prefix_audit_sha256=sha(OUT / 'completed-prefix-audit.json'),
        candidate_preparation_sha256=sha(OUT / 'candidate-preparation.json'),
        pair_manifest_sha256=sha(BUNDLE / 'pair-manifest.json'),
        recovered_stage_sha256={name: sha(path) for name, path in stages.items()},
        context='Actual persisted P16 merge status, original sample IDs, actual local payload and same immutable parent files; only head input discovery/hardware/artifact wiring adapted',
        actual_GPU_core_restarted=False, no_new_head_fitting=True, quality_read=False, C13_read=False, scoring_entries=0))
    p16 = read(working / 'p16_execution_receipt.json')
    sample = pd.read_csv(BASE / 'production-inputs/sample_submission.csv', dtype=str, keep_default_na=False)
    scope = dict(__name__='__main__', ROOT=working, INPUT=head_input, torch=torch, pd=pd, hashlib=hashlib, json=json,
        shutil=shutil, sys=sys, FROZEN_SHA256=p16['pipeline']['source_sha256'],
        MERGE_REPORT={'status': p16['merge_status']}, SAMPLE_IDS=sample.molecule_id.tolist(), IS_RERUN=p16['is_hidden_rerun'])
    assert not scope['IS_RERUN'] and scope['MERGE_REPORT']['status'] == 'applied'
    trials = {}
    for trial, path in stages.items():
        shutil.copyfile(working / 'c76_p16.csv', working / 'submission.csv')
        try: exec(compile(path.read_text(encoding='utf8'), 'c76_r1_' + trial + '_head', 'exec'), scope)
        except BaseException:
            (working / 'submission.csv').unlink(missing_ok=True); raise
        target = working / ('c76_' + trial + '_submission_r1.csv'); shutil.copyfile(working / 'submission.csv', target)
        receipt = read(working / ('c76_' + trial + '_execution_r1.json'))
        assert receipt['queries'] == 12 and receipt['valid_rows'] == 400 and receipt['exact_unprocessed_fallback_rows'] == 388
        assert receipt['trials'][trial]['changed_lists'] > 0
        trials[trial] = dict(csv_sha256=sha(target), changed_lists=receipt['trials'][trial]['changed_lists'],
            inactive_rows_preserved=388, receipt_sha256=sha(working / ('c76_' + trial + '_execution_r1.json')))
    # Verify that the interface repair is the identity on previously tested
    # all-active inputs, using the original exact old/cloud-tested vectors.
    old_work = Path(read(BASE / 'complete-pair-used-probe/results.json')['working'])
    used = OUT / 'used-active-pair'
    masked = REPO / 'outputs/local/c74-complete-clean-target-20261112/target-masked-reference'
    with (OUT / 'used-active-replay.log').open('w') as log:
        subprocess.run([sys.executable, str(BUNDLE / 'score_c76_complete_pair.py'), '--work', str(old_work),
            '--bundle', str(BUNDLE), '--reference', str(masked), '--output', str(used)],
            cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
    np.testing.assert_array_equal(np.load(used / 'features.npy'), np.load(old_work / 'c76_scores/features.npy'))
    for trial in ['cpu', 'gpu']:
        np.testing.assert_array_equal(np.load(used / (trial + '-scores.npy')), np.load(old_work / 'c76_scores' / (trial + '-scores.npy')))
        pd.testing.assert_frame_equal(pd.read_csv(used / (trial + '.csv'), dtype=str),
            pd.read_csv(old_work / 'c76_scores' / (trial + '.csv'), dtype=str), check_exact=True)
    # A failed ACTIVE row must still be rejected. Nothing is filtered by health.
    bad = OUT / 'active-failure-guard-input'; bad.mkdir()
    for name in ['c09e_external_receipt.json', 'c09e_p01_submission.csv', 'c76_p16.csv',
        'effective_query_spectra.parquet', 'execution_receipt.json', 'c09e_matches.parquet',
        'neural_public.parquet', 'neural_ours.parquet']:
        (bad / name).symlink_to((working / name).resolve())
    raw = pd.read_parquet(working / 'neural_candidate_features.parquet')
    active = set(pd.read_parquet(working / 'effective_query_spectra.parquet').molecule_id)
    index = raw.index[raw.molecule_id.isin(active)][0]; raw.loc[index, 'neural_ok'] = False
    raw.to_parquet(bad / 'neural_candidate_features.parquet', index=False)
    bad_output = OUT / 'active-failure-guard-output'
    with (OUT / 'active-failure-guard.log').open('w') as log:
        result = subprocess.run([sys.executable, str(BUNDLE / 'score_c76_complete_pair.py'), '--work', str(bad),
            '--bundle', str(BUNDLE), '--reference', str(reference), '--output', str(bad_output)],
            cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
    assert result.returncode != 0 and not bad_output.exists()
    assert 'assert frame.neural_ok.astype(bool).all()' in (OUT / 'active-failure-guard.log').read_text()
    for name, digest in prefix['reusable_unchanged_files_sha256'].items(): assert sha(working / name) == digest, name
    write(OUT / 'head-recovery-results.json', dict(completed_utc=stamp(), source_sha256=sha(__file__), passed=True,
        recovery_intent_sha256=sha(OUT / 'head-recovery-intent.json'),
        pair_manifest_sha256=sha(BUNDLE / 'pair-manifest.json'), working=str(working),
        original_full_run_passed=False, original_complete_core_and_P16_prefix_reused=True,
        GPU_core_restarted=False, actual_new_head_stages_executed=True,
        preview_queries=12, acquisitions=41, output_rows=400, inactive_rows=388, trials=trials,
        original_all_active_12query_51acquisition_vectors_scores_lists_bitexact=True,
        all_original_active_query_health_checks_preserved=True, corrupted_active_row_rejected=True,
        independent_physical_and_full_runtime_audit_still_required=True,
        no_new_head_fitting=True, quality_read=False, C13_read=False, scoring_entries=0, qualified_for_scoring=False))
    print('C76 R1 HEAD-ONLY RECOVERY PASSED;400ROWS/388FALLBACKS;OLD ALL-ACTIVE MATH BITEXACT', flush=True)


if __name__ == '__main__': main()
