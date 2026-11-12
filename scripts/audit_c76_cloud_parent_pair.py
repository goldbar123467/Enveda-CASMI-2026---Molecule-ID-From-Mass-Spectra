"""Require identical actual-T4 complete parents for the two fixed head trials."""
import os
for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']: os.environ[name] = '2'
import torch
import numpy as np
import pandas as pd
from c75_port_utils import ROOT, read, write, sha, stamp


def main():
    base = ROOT / 'outputs/local/c76-numerical-portability-20261112'
    target = base / 'actual-T4-production-parent-pair-audit.json'; assert not target.exists()
    folders = {trial: ROOT / 'outputs/forks' / ('c76-' + trial + '-production-v1') for trial in ['cpu', 'gpu']}
    audits = {}
    for trial, folder in folders.items():
        audit = read(folder / 'independent-production-audit.json')
        assert audit['execution_qualified'] and audit['trial'] == trial and audit['actual_two_T4s']
        assert audit['same_cloud_input_local_head_replay_exact'] and audit['remote_source_exact']
        assert audit['complete_pipeline_binding_sha256'] == sha(base / 'complete-pipeline-binding-r1.json')
        for name, pin in audit['artifact_sha256'].items(): assert sha(ROOT / name) == pin, name
        audits[trial] = sha(folder / 'independent-production-audit.json')
    comparisons = {}
    for name in ['c09e_p01_submission.csv', 'c76_p16.csv', 'c76_scores/physical-parent.csv']:
        a = pd.read_csv(folders['cpu'] / name, dtype=str, keep_default_na=False)
        b = pd.read_csv(folders['gpu'] / name, dtype=str, keep_default_na=False)
        pd.testing.assert_frame_equal(a, b, check_exact=True)
        comparisons[name] = dict(rows=len(a), exact=True,
            CPU_sha256=sha(folders['cpu'] / name), GPU_sha256=sha(folders['gpu'] / name))
    # Disclose any raw numerical variation; do not pass a failed old C75 gate.
    x = np.load(folders['cpu'] / 'c76_scores/features.npy')
    y = np.load(folders['gpu'] / 'c76_scores/features.npy')
    assert x.shape == y.shape
    write(target, dict(completed_utc=stamp(), source_sha256=sha(__file__), passed=True,
        complete_pipeline_binding_sha256=sha(base / 'complete-pipeline-binding-r1.json'),
        individual_actual_production_audits_sha256=audits,
        both_actual_two_T4_runs=True, both_same_cloud_input_local_head_replays_passed=True,
        original_P01_P16_and_physical_parents_exact_across_both_actual_T4_runs=True,
        parent_comparisons=comparisons, paired_head_features_bitexact=bool(np.array_equal(x, y)),
        paired_head_feature_maximum_absolute_difference=float(np.max(np.abs(x-y))),
        local_2070_full_core_equivalence_claimed=False, original_C75_gate_passed=False,
        C13_still_required=True, quality_labels_read=False, C13_read=False,
        scoring_entries=0, qualified_for_scoring=False))
    print('C76 ACTUAL T4 COMPLETE PARENT PAIR AUDIT PASSED', flush=True)


if __name__ == '__main__': main()
