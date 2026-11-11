"""Collect one exact completed C41 inference and verify runtime exclusions."""
import argparse,csv,hashlib,json,re,subprocess
from pathlib import Path
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scientific_sha(nb):
    cells=[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
    return hashlib.sha256(json.dumps(cells,ensure_ascii=True,separators=(',',':')).encode()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('slot');p.add_argument('--version',type=int,default=1);a=p.parse_args()
    assert re.fullmatch(r'c41(?:r[1-9]\d*)?-(?:smoke|full)-masked-validation',a.slot)
    from kaggle.api.kaggle_api_extended import KaggleApi,ApiGetKernelRequest
    api=KaggleApi();api.authenticate();kernel='clarkkitchen/casmi26-'+a.slot
    status=json.loads(str(api.kernels_status(kernel)));assert status['status']=='COMPLETE',status
    with api.build_kaggle_client() as client:
        req=ApiGetKernelRequest();req.user_name='clarkkitchen';req.kernel_slug=kernel.split('/')[1]
        response=json.loads(client.kernels.kernels_api_client.get_kernel(req).to_json())
    m=response['metadata'];assert m['currentVersionNumber']==a.version and m['isPrivate'] and not m['enableInternet']
    folder=ROOT/'kaggle/forks'/a.slot;pin=json.loads((folder/'validation-pin.json').read_text())
    assert sha(folder/'notebook.ipynb')==pin['notebook_sha256']
    remote=scientific_sha(json.loads(response['blob']['source']));assert remote==scientific_sha(json.loads((folder/'notebook.ipynb').read_text(encoding='utf8')))
    dest=ROOT/'outputs/forks'/(a.slot+'-v'+str(a.version));dest.mkdir(parents=True,exist_ok=True)
    assert not (dest/'collection-audit.json').exists(),'Already collected and audited'
    pattern=r'(c41_.*\.(json|csv|parquet|log)|neural_(public|ours)\.parquet|execution_receipt\.json|submission_pipeline\.log|casmi26-.*\.log)$'
    subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'scripts/fetch_output.py',a.slot,str(dest),pattern],cwd=ROOT,check=True,timeout=1800)
    required=['c41_report.json','c41_t4_allocation.json','c41_gpu1_preflight.json','c41_p01.csv','c41_p14.csv','c41_queries.parquet',
              'c41_main_reference_mask_audit.json','c41_engine2_reference_mask_receipt.json','c41_legacy_compatibility.json','execution_receipt.json']
    for name in required:assert (dest/name).is_file(),name
    report=json.loads((dest/'c41_report.json').read_text());n=pin['queries']
    assert report['queries']==n and report['baseline_replay_exact'] and not report['metrics_computed'] and not report['query_labels_mounted']
    assert not report['scoring_submission'] and report['validation_contract_sha256']==pin['query_contract_sha256']
    assert sha(dest/'c41_p01.csv')==report['baseline_sha256']
    allocation=json.loads((dest/'c41_t4_allocation.json').read_text())
    assert len(allocation['devices'])==2 and all('T4' in d['name'] for d in allocation['devices'])
    for name in ['c41_main_reference_mask_audit.json','c41_engine2_reference_mask_receipt.json']:
        audit=json.loads((dest/name).read_text());assert audit['excluded_reference_rows']>0 and audit['forbidden_rows_retained']==0
        if 'main' in name:assert audit['window_calls']>0 and audit['representative_calls']>0
    execution=json.loads((dest/'execution_receipt.json').read_text())
    assert execution['is_full_run'] and execution['processed_molecules']==n
    assert report['p14_receipt']['input_errors']==0
    compat=json.loads((dest/'c41_legacy_compatibility.json').read_text())
    assert set(compat)=={'v3','v4'}
    for row in compat.values():
        assert set(row['before_sha256'])==set(row['after_sha256'])=={'chem.py','frag.py'}
        assert all(row['before_sha256'][k]!=row['after_sha256'][k] for k in row['before_sha256'])
    for name in ['c41_p01.csv','c41_p14.csv']:
        with (dest/name).open(newline='',encoding='utf8') as stream:rows=list(csv.DictReader(stream))
        assert len(rows)==n and len({r['molecule_id'] for r in rows})==n
        assert all(1<=len(r['smiles'].split(';'))<=25 for r in rows)
    result=dict(completed_utc=datetime.now(timezone.utc).isoformat(),kernel=kernel,version=a.version,queries=n,
        actual_two_T4s=True,main_and_engine2_exclusions_enforced=True,forbidden_reference_rows=0,
        formate_source_copy_verified=True,baseline_replay_exact=True,remote_source_exact=True,
        remote_scientific_sha256=remote,validation_pin_sha256=sha(folder/'validation-pin.json'),
        files_sha256={name:sha(dest/name) for name in required},quality_metrics_read=False,C13_read=False,scoring_submissions=0)
    (dest/'collection-audit.json').write_text(json.dumps(result,indent=2));print('C41 COLLECTION PASS',json.dumps(result),flush=True)


if __name__=='__main__':main()
