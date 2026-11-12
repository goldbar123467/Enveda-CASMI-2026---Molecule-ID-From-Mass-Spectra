"""Audit a completed C42 production smoke, including independent local replay."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
import argparse,json,subprocess,sys
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from precision_cache import ROOT,sha,write,features
from submit_c09_external_once import scientific_sha


def main():
    p=argparse.ArgumentParser();p.add_argument('trial',choices=['cpu','gpu']);a=p.parse_args()
    trial=a.trial;slot='p20-c42-cpu-alignment' if trial=='cpu' else 'p21-c42-gpu-alignment'
    folder=ROOT/'kaggle/forks'/slot;pin=json.loads((folder/'production-pin.json').read_text())
    dest=ROOT/'outputs/forks'/(slot+'-v1');dest.mkdir(parents=True,exist_ok=True)
    target=dest/'independent-production-audit.json';assert not target.exists()
    assert sha(folder/'notebook.ipynb')==pin['notebook_sha256']
    from kaggle.api.kaggle_api_extended import KaggleApi,ApiGetKernelRequest
    api=KaggleApi();api.authenticate();kernel='clarkkitchen/casmi26-'+slot
    state=json.loads(str(api.kernels_status(kernel)));assert state['status']=='COMPLETE',state
    with api.build_kaggle_client() as client:
        request=ApiGetKernelRequest();request.user_name='clarkkitchen';request.kernel_slug=kernel.split('/')[1]
        remote=json.loads(client.kernels.kernels_api_client.get_kernel(request).to_json())
    m=remote['metadata'];assert m['currentVersionNumber']==1 and m['isPrivate'] and not m['enableInternet']
    remote_sha=scientific_sha(json.loads(remote['blob']['source']));assert remote_sha==pin['scientific_sha256']
    pattern=r'(c42_.*\.(csv|json|parquet|log)|c42_scores/.*\.(csv|parquet|json|npz)|execution_receipt\.json|p16_execution_receipt\.json|neural_(public|ours)\.parquet|neural_candidate_features\.parquet|effective_query_spectra\.parquet|submission\.csv)$'
    subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'scripts/fetch_output.py',slot,str(dest),pattern],cwd=ROOT,check=True,timeout=1800)
    r=json.loads((dest/'c42_execution.json').read_text())
    assert r['trial']==trial and not r['is_hidden_rerun'] and r['valid_rows']==400 and r['actual_query_count']==12
    assert not r['query_labels_mounted'] and not r['training_performed'] and not r['validation_reference_masks_mounted']
    assert r['pipeline']['processed_molecules']==12 and not r['pipeline']['is_full_run']
    assert r['exact_unprocessed_fallback_rows']==388
    assert len(r['devices'])==2 and all('T4' in d['name'] for d in r['devices'])
    head=ROOT/'outputs/publish/c42r1-crossmodal-heads-20261111'
    assert r['head_manifest_sha256']==pin['head_manifest_sha256']==sha(head/'head-manifest.json')
    assert sha(dest/'submission.csv')==r['csv_sha256'] and sha(dest/'c42_parent.csv')==r['parent_csv_sha256']
    p16=json.loads((dest/'p16_execution_receipt.json').read_text());assert p16['csv_sha256']==r['parent_csv_sha256']
    previous=ROOT/'outputs/forks/p19-c18-guarded-export-v1/past_trials_parent.csv'
    assert sha(previous)==r['parent_csv_sha256'],'Preview parent changed; diagnose before qualifying the new component'
    parent=pd.read_csv(dest/'c42_parent.csv',dtype=str,keep_default_na=False);answer=pd.read_csv(dest/'submission.csv',dtype=str,keep_default_na=False)
    assert parent.molecule_id.tolist()==answer.molecule_id.tolist() and answer.molecule_id.is_unique
    queries=pd.read_parquet(dest/'effective_query_spectra.parquet');active=set(queries.molecule_id.astype(str))
    assert len(active)==12 and answer.loc[~answer.molecule_id.isin(active)].equals(parent.loc[~parent.molecule_id.isin(active)])
    identities={};changed=0
    for old,new in zip(parent.itertuples(index=False),answer.itertuples(index=False)):
        before,after=old.smiles.split(';'),new.smiles.split(';')
        assert before[0]==after[0] and set(before)==set(after) and len(before)==len(after)
        for s in before:
            if s not in identities:
                key,_,formula,*_=features.structure(s);identities[s]=(key,formula)
        keys=[identities[s][0] for s in after];assert len(set(keys))==len(keys)
        assert [identities[s][1] for s in before]==[identities[s][1] for s in after]
        changed+=before!=after
    scores=dest/'c42_scores';receipt=json.loads((scores/'score-receipt.json').read_text())
    assert receipt==r['score_receipt'] and receipt['trials'][trial]['changed_lists']==changed and changed>0
    if not (scores/'independent-ranking-audit.json').exists():
        subprocess.run([str(ROOT/'.venv-gpu/Scripts/python.exe'),'scripts/audit_c42_ranking.py',str(scores),str(dest/'c42_active_parent.csv')],cwd=ROOT,check=True)
    local=dest/'local-head-replay'
    if not (local/'score-receipt.json').exists():
        assert not local.exists(),'Partial local replay; inspect it before recovery'
        with (dest/'local-head-replay.log').open('w') as log:
            subprocess.run([str(ROOT/'.venv-encoders/Scripts/python.exe'),str(head/'score_c42_crossmodal.py'),str(head),
                str(ROOT/'outputs/publish/frozen-p14-20261107'),str(dest/'c42_active_parent.csv'),str(dest/'effective_query_spectra.parquet'),
                str(local),'--only',trial],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,check=True)
    cloud_score=pd.read_parquet(scores/'candidate-scores.parquet').set_index(['trial','molecule_id','parent_rank','smiles'])
    local_score=pd.read_parquet(local/'candidate-scores.parquet').set_index(cloud_score.index.names).reindex(cloud_score.index)
    assert cloud_score.eligible.equals(local_score.eligible)
    error=float(np.abs(cloud_score.score-local_score.score).max());assert np.isfinite(error) and error<.005,error
    local_answer=pd.read_csv(local/(trial+'.csv'),dtype=str,keep_default_na=False)
    cloud_answer=pd.read_csv(scores/(trial+'.csv'),dtype=str,keep_default_na=False)
    assert local_answer.equals(cloud_answer),'Device difference changes final rankings'
    public=pd.read_parquet(dest/'neural_public.parquet').set_index(['molecule_id','smiles'])
    ours=pd.read_parquet(dest/'neural_ours.parquet').set_index(public.index.names).reindex(public.index)
    blended=pd.read_parquet(dest/'neural_candidate_features.parquet').set_index(public.index.names).reindex(public.index)
    def z(frame):
        s=frame.neural_score.fillna(0);g=s.groupby(level=0);return (s-g.transform('mean'))/g.transform(lambda v:v.std(ddof=0)).clip(lower=1e-6)
    parent_error=float(np.abs(blended.neural_score-.5*z(public)-.5*z(ours)).max());assert parent_error<1e-6
    result=dict(completed_utc=datetime.now(timezone.utc).isoformat(),trial=trial,execution_qualified=True,
        source_sha256=sha(__file__),remote_source_exact=True,remote_scientific_sha256=remote_sha,notebook_sha256=pin['notebook_sha256'],
        head_manifest_sha256=pin['head_manifest_sha256'],csv_sha256=r['csv_sha256'],valid_rows=400,real_preview_queries=12,
        changed_real_preview_lists=int(changed),exact_unprocessed_fallback_rows=388,rank1_changes=0,candidate_membership_changes=0,
        formula_slot_changes=0,official_identity_duplicates=0,independent_tail_arithmetic_exact=True,
        local_cloud_final_lists_exact=True,local_cloud_score_max_abs_difference=error,unchanged_P01_blend_max_abs_error=parent_error,
        parent_preview_exact_previous=True,quality_labels_read=False,C13_read=False,statistical_promotion=False,
        artifact_sha256={str(p.relative_to(ROOT)):sha(p) for p in [dest/'c42_execution.json',dest/'submission.csv',
            scores/'score-receipt.json',scores/'independent-ranking-audit.json',local/'score-receipt.json',folder/'production-pin.json']})
    write(target,result);print('C42 PRODUCTION AUDIT PASS',json.dumps(result),flush=True)


if __name__=='__main__':main()
