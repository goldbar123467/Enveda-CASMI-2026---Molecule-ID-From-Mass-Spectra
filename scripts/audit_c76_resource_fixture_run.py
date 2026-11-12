"""Audit actual C76 complete execution and acquisition/source invariants.

This is the explicitly revised numerical candidate's engineering audit. It
never changes or passes the prior C75 exact-cloud reproduction gate, and never
reads truth keys or computes quality. N1's real CPU synthetic path is separate.
"""
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[name]='2'
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from functools import lru_cache
from c75_port_utils import ROOT,read,write,sha,stamp
from audit_complete_core_health import check_events


def main():
    base=ROOT/'outputs/local/c76-numerical-portability-20261112';port=base/'fixture-resource-port'
    dest=port/'independent-run-audit.json';assert not dest.exists()
    pin=read(port/'fixture-freeze.json');done=read(port/'run-complete.json');working=Path(done['working'])
    assert read(port/'supervisor-complete.json')['status']=='complete'
    assert done['source_sha256']==sha(ROOT/'scripts/run_c76_resource_fixture.py') and done['queries']==12 and done['acquisitions']==51
    assert sha(port/'notebook.ipynb')==pin['notebook_sha256']
    source_audit=read(port/'independent-source-audit.json');assert source_audit['passed']
    assert source_audit['source_sha256']==sha(ROOT/'scripts/audit_c76_resource_source.py')
    assert source_audit['fixture_freeze_sha256']==sha(port/'fixture-freeze.json')
    devices=read(working/'c75_gpu_allocation.json');assert devices['local_single_GPU'] and not devices['actual_two_T4s']
    assert len(devices['devices'])==1 and '2070' in devices['devices'][0]['name']
    preflight=read(working/'c41_gpu1_preflight.json');assert preflight['dreams_finite'] and preflight['molformer_finite']
    logs=[port/'pipeline.log',working/'submission_pipeline.log']
    health=check_events([dict(data=p.read_text(encoding='utf8',errors='replace')) for p in logs],12)
    execution=read(working/'execution_receipt.json');assert execution['is_full_run'] and execution['processed_molecules']==12
    report=read(working/'c41_report.json');assert report['queries']==12 and report['baseline_replay_exact'] and report['p14_receipt']['input_errors']==0
    assert not report['metrics_computed'] and not report['query_labels_mounted']
    for name in ['c41_main_reference_mask_audit.json','c41_engine2_reference_mask_receipt.json']:
        receipt=read(working/name);assert receipt['excluded_reference_rows']>0 and receipt['forbidden_rows_retained']==0
        if 'main' in name:assert receipt['window_calls']>0 and receipt['engine_representative_builds']>0 and receipt['engine_representatives_removed']>0
    expected=pd.read_parquet(Path(pin['query_package'])/'validation-queries.parquet')
    actual=pd.read_parquet(working/'c41_queries.parquet')
    assert len(actual)==len(expected)==51 and actual.molecule_id.nunique()==expected.molecule_id.nunique()==12
    assert actual.spectrum_id.is_unique and set(actual.spectrum_id)==set(expected.spectrum_id)
    aa=actual.set_index('spectrum_id').loc[expected.spectrum_id];zz=expected.set_index('spectrum_id')
    for col in zz.columns:
        for x,y in zip(aa[col],zz[col]):
            if isinstance(y,(np.ndarray,list,tuple)):np.testing.assert_array_equal(np.asarray(x),np.asarray(y),err_msg=col)
            elif pd.isna(y):assert pd.isna(x),col
            else:assert x==y,col
    from c74_complete_target import module
    identify=lru_cache(maxsize=10000)(module('c76_fixture_identity','c09e_features.py').structure)
    lists={};previous=Path('/home/imsa/.local/share/enveda-c75/fixture/working')
    for name in ['c41_p01.csv','c41_p14.csv']:
        frame=pd.read_csv(working/name,dtype=str,keep_default_na=False);old=pd.read_csv(previous/name,dtype=str,keep_default_na=False)
        assert frame.molecule_id.is_unique and frame.molecule_id.tolist()==list(dict.fromkeys(expected.molecule_id))
        assert frame.molecule_id.tolist()==old.molecule_id.tolist()
        for line in frame.smiles:
            values=line.split(';');assert 1<=len(values)<=25
            keys=[identify(s)[0] for s in values];assert all(keys) and len(set(keys))==len(keys)
        lists[name]=dict(valid_rows=len(frame),changed_from_old_local=int(frame.smiles.ne(old.smiles).sum()),
            leader_changes_from_old_local=sum(a.split(';')[0]!=z.split(';')[0] for a,z in zip(frame.smiles,old.smiles)))
    for name in ['c41_neural_p01.parquet','neural_public.parquet','neural_ours.parquet']:
        frame=pd.read_parquet(working/name);assert frame.molecule_id.nunique()==12 and frame.neural_ok.astype(bool).all()
        assert not frame.duplicated(['molecule_id','metric_key']).any() and np.isfinite(frame.neural_score).all()
    for name in ['gl_work/gl_out.json.meta.json','ice_work/ice_out.json.meta.json']:
        meta=read(working/name);assert meta['status']=='ok' and meta['n_mols_scored']==meta['n_mols_covered']==9
        assert meta['n_pred_fail']==0 and meta.get('n_feat_fail',0)==0
    adapter=read(working/'gl_work/gl_out.json.c76_adapter.json')
    assert adapter['weights_exact'] and adapter['decision_device']=='cpu' and adapter['cpu_threads']==2
    assert adapter['intensity_device'].startswith('cuda')
    component=read(base/'extended-used-nine-r1/results.json');assert component['component_passed'] and component['comparison']['finite_scores']==799
    cpu=read(base/'cpu-feature-parity/independent-comparison.json');assert len(cpu['array_comparisons'])==26 and all(v['exact'] for v in cpu['array_comparisons'].values())
    proxy=read(base/'n1-resource-path-probe/results.json');assert proxy['passed'] and proxy['finite_outputs']==32 and proxy['synthetic_only']
    assert proxy['source_sha256']==sha(ROOT/'scripts/probe_c76_resource_n1_path.py')
    assert not (ROOT/'outputs/local/c72-clean-target-holdout-20261112/evaluation-consumption.json').exists()
    assert not (ROOT/'outputs/local/c13-confirmation-20261110/evaluation-consumption.json').exists()
    files=[port/'fixture-freeze.json',port/'independent-source-audit.json',port/'run-complete.json',
        working/'c41_p01.csv',working/'c41_p14.csv',working/'c41_report.json',working/'execution_receipt.json',
        working/'gl_work/gl_out.json.c76_adapter.json',base/'n1-resource-path-probe/results.json',base/'extended-used-nine-r1/results.json']
    write(dest,dict(completed_utc=stamp(),source_sha256=sha(__file__),fixture_freeze_sha256=sha(port/'fixture-freeze.json'),
        actual_complete_single_2070_run=True,queries=12,acquisitions=51,all_original_metadata_and_peaks_exact=True,
        local_resource_policy_sha256=pin['operational_resource_policy_sha256'],ICE_maximum_batch_size=8,
        exact_ICE_parity_across_batch_sizes_not_assumed=True,
        full_core_health_passed=health,reference_exclusions_passed=True,lists=lists,
        actual_classic_GL_CPU_decision_GPU_intensity=True,real_original_N1_CPU_proxy_and_merged_CE_exercise=True,
        original_C75_exact_cloud_reproduction_gate_passed=False,C76_engineering_fixture_passed=True,
        fresh_source_release_requires_complete_C76_binding=True,actual_T4_production_smoke_still_required=True,
        statistical_qualification=False,qualified_for_scoring=False,quality_read=False,C13_read=False,scoring_entries=0,
        files_sha256={str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):sha(p) for p in files}))
    print('C76 COMPLETE ENGINEERING FIXTURE AUDIT PASSED; SOURCE/CONFIRMATION/PRODUCTION STILL HELD',flush=True)


if __name__=='__main__':main()
