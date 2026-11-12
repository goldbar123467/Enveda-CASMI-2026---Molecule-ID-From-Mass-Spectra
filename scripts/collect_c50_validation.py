"""Collect the exact completed auxiliary core; no labels or quality metrics."""
import csv,json,subprocess,sys
from datetime import datetime,timezone
from collect_c41_validation import ROOT,sha,scientific_sha


def main():
    from kaggle.api.kaggle_api_extended import KaggleApi,ApiGetKernelRequest
    slot='c50-full-masked-validation';kernel='clarkkitchen/casmi26-'+slot
    folder=ROOT/'kaggle/forks'/slot;pin=json.loads((folder/'validation-pin.json').read_text())
    dest=ROOT/'outputs/forks'/(slot+'-v1');dest.mkdir(parents=True,exist_ok=True);assert not (dest/'collection-audit.json').exists()
    api=KaggleApi();api.authenticate();state=json.loads(str(api.kernels_status(kernel)));assert state['status']=='COMPLETE',state
    with api.build_kaggle_client() as client:
        request=ApiGetKernelRequest();request.user_name='clarkkitchen';request.kernel_slug=kernel.split('/')[1]
        response=json.loads(client.kernels.kernels_api_client.get_kernel(request).to_json())
    metadata=response['metadata'];assert metadata['currentVersionNumber']==1 and metadata['isPrivate'] and not metadata['enableInternet']
    assert sha(folder/'notebook.ipynb')==pin['notebook_sha256']
    remote=scientific_sha(json.loads(response['blob']['source']));assert remote==scientific_sha(json.loads((folder/'notebook.ipynb').read_text(encoding='utf8')))
    pattern=r'(c41_.*\.(json|csv|parquet|log)|neural_(public|ours)\.parquet|execution_receipt\.json|frozen_candidate_evidence\.parquet|neural_candidate_features\.parquet|outsider_neural_features\.parquet|effective_query_spectra\.parquet|submission_pipeline\.log)$'
    subprocess.run([sys.executable,'scripts/fetch_output.py',slot,str(dest),pattern],cwd=ROOT,check=True,timeout=1800)
    required=['c41_report.json','c41_t4_allocation.json','c41_gpu1_preflight.json','c41_p01.csv','c41_p14.csv','c41_queries.parquet',
        'c41_neural_p01.parquet','frozen_candidate_evidence.parquet','outsider_neural_features.parquet',
        'c41_main_reference_mask_audit.json','c41_engine2_reference_mask_receipt.json','c41_legacy_compatibility.json','execution_receipt.json']
    for name in required:assert (dest/name).is_file(),name
    report=json.loads((dest/'c41_report.json').read_text());assert report['queries']==35 and report['baseline_replay_exact']
    assert not report['metrics_computed'] and not report['query_labels_mounted'] and not report['scoring_submission']
    assert report['validation_contract_sha256']==pin['query_contract_sha256'] and sha(dest/'c41_p01.csv')==report['baseline_sha256']
    allocation=json.loads((dest/'c41_t4_allocation.json').read_text());assert len(allocation['devices'])==2 and all('T4' in row['name'] for row in allocation['devices'])
    for name in ['c41_main_reference_mask_audit.json','c41_engine2_reference_mask_receipt.json']:
        value=json.loads((dest/name).read_text());assert value['excluded_reference_rows']>0 and value['forbidden_rows_retained']==0
        if 'main' in name:assert value['window_calls']>0 and value['engine_representative_builds']>0 and value['engine_representatives_removed']>0
    execution=json.loads((dest/'execution_receipt.json').read_text());assert execution['is_full_run'] and execution['processed_molecules']==35
    assert report['p14_receipt']['input_errors']==0
    compatibility=json.loads((dest/'c41_legacy_compatibility.json').read_text());assert set(compatibility)=={'v3','v4'}
    for value in compatibility.values():
        assert set(value['before_sha256'])==set(value['after_sha256'])=={'chem.py','frag.py'}
        assert all(value['before_sha256'][name]!=value['after_sha256'][name] for name in value['before_sha256'])
    for name in ['c41_p01.csv','c41_p14.csv']:
        with (dest/name).open(newline='',encoding='utf8') as stream:rows=list(csv.DictReader(stream))
        assert len(rows)==len({row['molecule_id'] for row in rows})==35 and all(1<=len(row['smiles'].split(';'))<=25 for row in rows)
    result=dict(completed_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),kernel=kernel,version=1,queries=35,
        actual_two_T4s=True,remote_source_exact=True,remote_scientific_sha256=remote,main_and_engine2_exclusions_enforced=True,
        forbidden_reference_rows=0,baseline_replay_exact=True,formate_source_copy_verified=True,
        validation_pin_sha256=sha(folder/'validation-pin.json'),files_sha256={name:sha(dest/name) for name in required},
        auxiliary_core_only=True,P16_still_requires_separate_complete_stage_replay=True,quality_metrics_read=False,C13_read=False,scoring_submissions=0)
    (dest/'collection-audit.json').write_text(json.dumps(result,indent=2),encoding='utf8');print('C50 BASE COLLECTION PASS',json.dumps(result))


if __name__=='__main__':main()
