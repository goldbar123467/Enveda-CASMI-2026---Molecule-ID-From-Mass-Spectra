"""Collect the completed faithful120-query core without confirmation outcomes."""
import csv,json,subprocess,sys
from release_c59_c13 import ROOT,OUT,FOLDER,KERNEL,read,write,sha,stamp
from submit_c09_external_once import scientific_sha


def main():
    from kaggle.api.kaggle_api_extended import KaggleApi,ApiGetKernelRequest
    assert not (ROOT/'outputs/local/c13-confirmation-20261110/evaluation-consumption.json').exists()
    pin=read(FOLDER/'compatibility-pin.json');push=read(OUT/'kernel-push-result.json')
    assert push['version']==1 and push['kernel']==KERNEL and push['notebook_sha256']==pin['notebook_sha256']
    assert sha(FOLDER/'notebook.ipynb')==pin['notebook_sha256']
    dest=ROOT/'outputs/forks/c59-c13-main-compatibility-v1';dest.mkdir(exist_ok=True);assert not (dest/'collection-audit.json').exists()
    api=KaggleApi();api.authenticate();state=json.loads(str(api.kernels_status(KERNEL)));assert state['status']=='COMPLETE',state
    with api.build_kaggle_client() as client:
        request=ApiGetKernelRequest();request.user_name='clarkkitchen';request.kernel_slug=KERNEL.split('/')[1]
        remote=json.loads(client.kernels.kernels_api_client.get_kernel(request).to_json())
    m=remote['metadata'];assert m['currentVersionNumber']==1 and m['isPrivate'] and not m['enableInternet']
    assert scientific_sha(json.loads(remote['blob']['source']))==pin['scientific_sha256']
    pattern=r'(c13_.*\.(json|csv|parquet|log)|c59_legacy_compatibility\.json|execution_receipt\.json|pc_lists\.json|eng_lists\.json|frozen_candidate_evidence\.parquet|neural_(public|ours|candidate_features)\.parquet|outsider_neural_features\.parquet|effective_query_spectra\.parquet|submission_pipeline\.log)$'
    subprocess.run([sys.executable,'scripts/fetch_output.py','c59-c13-main-compatibility',str(dest),pattern],cwd=ROOT,check=True,timeout=1800)
    required=['c13_report.json','c13_t4_allocation.json','c13_gpu1_preflight.json','c13_p01.csv','c13_p14.csv','c13_queries.parquet',
        'c13_neural_p01.parquet','frozen_candidate_evidence.parquet','outsider_neural_features.parquet','c59_legacy_compatibility.json','execution_receipt.json','pc_lists.json','eng_lists.json']
    for name in required:assert (dest/name).exists(),name
    report=read(dest/'c13_report.json');assert report['queries']==120 and report['baseline_replay_exact']
    assert not report['metrics_computed'] and not report['query_labels_mounted'] and not report['recipe_selection']
    assert report['validation_contract_sha256']==pin['query_contract_sha256'] and report['baseline_sha256']==sha(dest/'c13_p01.csv')
    assert report['p14_receipt']['input_errors']==0
    allocation=read(dest/'c13_t4_allocation.json');assert len(allocation['devices'])==2 and all('T4' in d['name'] for d in allocation['devices'])
    execution=read(dest/'execution_receipt.json');assert execution['is_full_run'] and execution['processed_molecules']==120
    assert execution['source_sha256']==pin['frozen_scientific_payload_sha256']
    compat=read(dest/'c59_legacy_compatibility.json');assert set(compat)=={'v3','v4'}
    from legacy_formate_compat import patch_chem_source,patch_frag_source
    import hashlib
    original=ROOT/'outputs/local/c37-selection-trace-20261111/pinned-public-source'
    before={n:sha(original/n) for n in ['chem.py','frag.py']}
    after={n:hashlib.sha256(fn((original/n).read_text(encoding='utf8')).encode()).hexdigest() for n,fn in [('chem.py',patch_chem_source),('frag.py',patch_frag_source)]}
    for value in compat.values():assert value['before_sha256']==before and value['after_sha256']==after
    for name in ['c13_p01.csv','c13_p14.csv']:
        with (dest/name).open(newline='',encoding='utf8') as stream:rows=list(csv.DictReader(stream))
        assert len(rows)==len({r['molecule_id'] for r in rows})==120 and all(1<=len(r['smiles'].split(';'))<=25 for r in rows)
    write(dest/'collection-audit.json',dict(completed_utc=stamp(),source_sha256=sha(__file__),kernel=KERNEL,version=1,queries=120,
        remote_source_exact=True,remote_scientific_sha256=pin['scientific_sha256'],actual_two_T4s=True,baseline_replay_exact=True,
        actual_main_fragment_sources_exact_production_adapter=True,PV_v2repair_unchanged=True,compatibility_pin_sha256=sha(FOLDER/'compatibility-pin.json'),
        files_sha256={name:sha(dest/name) for name in required},quality_outcomes_read=False,C13_consumed=False,scoring_submissions=0,
        P16_complete_stage_still_required=True))
    print('C59 FAITHFUL UNLABELLED CORE COLLECTED',flush=True)


if __name__=='__main__':main()
