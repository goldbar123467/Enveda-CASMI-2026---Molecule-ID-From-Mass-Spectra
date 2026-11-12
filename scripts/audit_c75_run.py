"""Audit complete local execution, original acquisition fidelity and USED parity.

Never reads a query truth key or computes quality metrics. Fixture acceptance is
exact final P01/P14 lists, exact feature keys/nonfloat values/NaN patterns and
numeric agreement at predeclared atol=rtol=1e-5 against completed cloud C57.
"""
import os
for n in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(n,'2')
import torch
import argparse,json
import numpy as np
import pandas as pd
from c75_port_utils import ROOT,read,write,sha,stamp
from audit_complete_core_health import check_events
OUT=ROOT/'outputs/local/c75-local-core-port-20261112'


def compare_frame(local,expected):
    keys=['molecule_id','metric_key'];assert not local.duplicated(keys).any() and not expected.duplicated(keys).any()
    a=local.set_index(keys).sort_index();b=expected.set_index(keys).sort_index();assert a.index.equals(b.index)
    assert set(a.columns)==set(b.columns)
    maxima={}
    for c in a.columns:
        if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c]):
            u=a[c].to_numpy(float);v=b[c].to_numpy(float);np.testing.assert_array_equal(np.isnan(u),np.isnan(v))
            if pd.api.types.is_float_dtype(a[c]) or pd.api.types.is_float_dtype(b[c]):
                np.testing.assert_allclose(u,v,atol=1e-5,rtol=1e-5,equal_nan=True,err_msg=c)
                finite=np.isfinite(u)&np.isfinite(v);maxima[c]=float(np.abs(u[finite]-v[finite]).max(initial=0))
            else:np.testing.assert_array_equal(u,v,err_msg=c)
        else:assert a[c].fillna('MISSING').tolist()==b[c].fillna('MISSING').tolist(),c
    return maxima


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['fixture','full']);args=parser.parse_args();mode=args.mode
    port=OUT/(mode+'-port');pin=read(port/'port-freeze.json');done=read(port/'run-complete.json')
    working=__import__('pathlib').Path(done['working']);dest=port/'independent-run-audit.json';assert not dest.exists()
    assert sha(port/'notebook.ipynb')==pin['local_notebook_sha256']
    assert done['source_sha256']==sha(ROOT/'scripts/run_c75_local_port.py')
    assert done['queries']==pin['queries'] and done['acquisitions']==pin['acquisitions']
    assert done['actual_local_single_2070'] and not done['actual_two_T4s']
    actual=read(working/'c75_gpu_allocation.json');assert actual['local_single_GPU'] and not actual['actual_two_T4s']
    assert len(actual['devices'])==1 and '2070' in actual['devices'][0]['name']
    preflight=read(working/'c41_gpu1_preflight.json')
    assert preflight['physical_device_requested']=='0' and '2070' in preflight['name'] and preflight['dreams_finite'] and preflight['molformer_finite']
    report=read(working/'c41_report.json');execution=read(working/'execution_receipt.json')
    assert report['queries']==pin['queries'] and report['baseline_replay_exact'] and report['p14_receipt']['input_errors']==0
    assert not report['metrics_computed'] and not report['query_labels_mounted'] and not report['scoring_submission']
    assert execution['is_full_run'] and execution['processed_molecules']==pin['queries']
    # Stage summaries are written by the outer orchestrator; detailed channel
    # counts are in the nested scientific runner. Audit both actual log files.
    outer_log=port/'local-pipeline.log';inner_log=working/'submission_pipeline.log'
    health=check_events([dict(data=p.read_text(encoding='utf8',errors='replace')) for p in [outer_log,inner_log]],pin['queries'])
    for name in ['c41_main_reference_mask_audit.json','c41_engine2_reference_mask_receipt.json']:
        receipt=read(working/name);assert receipt['excluded_reference_rows']>0 and receipt['forbidden_rows_retained']==0
        if 'main' in name:assert receipt['window_calls']>0 and receipt['engine_representative_builds']>0 and receipt['engine_representatives_removed']>0
    expected_q=pd.read_parquet(port/'query-package/validation-queries.parquet');q=pd.read_parquet(working/'c41_queries.parquet')
    assert q.spectrum_id.is_unique and set(q.spectrum_id)==set(expected_q.spectrum_id) and len(q)==pin['acquisitions']
    a=q.set_index('spectrum_id').loc[expected_q.spectrum_id];b=expected_q.set_index('spectrum_id')
    for c in b.columns:
        for u,v in zip(a[c],b[c]):
            if isinstance(v,(np.ndarray,list,tuple)):np.testing.assert_array_equal(np.asarray(u),np.asarray(v),err_msg=c)
            elif pd.isna(v):assert pd.isna(u),c
            else:assert u==v,c
    assert set(q.molecule_id)==set(expected_q.molecule_id) and q.molecule_id.nunique()==pin['queries']
    for name in ['c41_p01.csv','c41_p14.csv']:
        f=pd.read_csv(working/name,dtype=str);assert f.molecule_id.is_unique and f.molecule_id.tolist()==list(dict.fromkeys(expected_q.molecule_id))
        assert f.smiles.str.split(';').map(len).between(1,25).all()
    for name in ['c41_neural_p01.parquet','neural_public.parquet','neural_ours.parquet']:
        f=pd.read_parquet(working/name);assert f.molecule_id.nunique()==pin['queries'] and f.neural_ok.astype(bool).all()
        assert np.isfinite(f.neural_score.to_numpy()).all()
    comparisons={}
    if mode=='fixture':
        cloud=ROOT/'outputs/forks/c57-full-masked-validation-v1';assert read(cloud/'independent-core-health-audit.json')['passed']
        ids=set(expected_q.molecule_id)
        for name in ['c41_p01.csv','c41_p14.csv']:
            ours=pd.read_csv(working/name,dtype=str).set_index('molecule_id');old=pd.read_csv(cloud/name,dtype=str).set_index('molecule_id')
            pd.testing.assert_frame_equal(ours,old.loc[ours.index],check_exact=True)
        for name in ['c41_neural_p01.parquet','neural_public.parquet','neural_ours.parquet']:
            ours=pd.read_parquet(working/name);old=pd.read_parquet(cloud/name);old=old[old.molecule_id.isin(ids)]
            comparisons[name]=compare_frame(ours,old)
    write(dest,dict(completed_utc=stamp(),source_sha256=sha(__file__),run_complete_sha256=sha(port/'run-complete.json'),mode=mode,
        queries=pin['queries'],acquisitions=pin['acquisitions'],actual_single_RTX2070=True,actual_two_T4s=False,
        full_original_acquisitions_metadata_peaks_exact=True,no_core_soft_failure=True,all_reference_guards_forbidden_rows_zero=True,
        all_neural_rows_healthy=True,**health,actual_health_log_files_sha256={'outer':sha(outer_log),'inner':sha(inner_log)},numeric_fixture_tolerance=dict(atol=1e-5,rtol=1e-5),fixture_feature_maximum_abs_differences=comparisons,
        fixture_P01_and_P14_lists_exact=(mode=='fixture'),source_and_runtime_parity_passed=(mode=='fixture'),
        query_truth_labels_or_quality_read=False,new_source_quality_read=False,C13_read=False,scoring_entries=0,passed=True))
    if mode=='fixture':
        write(OUT/'fixture-parity-audit.json',dict(completed_utc=stamp(),source_sha256=sha(__file__),independent_run_audit_sha256=sha(dest),
            used_queries=pin['queries'],original_acquisitions=pin['acquisitions'],original_C57_reference_masks=True,
            full_P01_P14_lists_exact=True,all_feature_rows_match=True,numeric_tolerance=dict(atol=1e-5,rtol=1e-5),
            actual_single_RTX2070=True,actual_two_T4s=False,source_and_full_flow_audited=True,quality_outcomes_read=False,C13_read=False,passed=True))
    print('C75 INDEPENDENT FULL RUN AUDIT PASS',mode,'parity',mode=='fixture',flush=True)


if __name__=='__main__':main()
