"""Audit a completed C49 production smoke, including independent local replay."""
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
    trial=a.trial;family=json.loads((ROOT/'outputs/local/c49-consensus-pair-20261112/complete-candidate-freeze.json').read_text());entry=family['candidates'][trial];slot=__import__('pathlib').Path(entry['path']).name
    folder=ROOT/entry['path'];pin=json.loads((folder/'production-pin.json').read_text());assert sha(folder/'production-pin.json')==entry['pin_sha256']
    dest=ROOT/'outputs/forks'/(slot+'-v1');dest.mkdir(parents=True,exist_ok=True)
    target=dest/'independent-production-audit.json';assert not target.exists()
    assert sha(folder/'notebook.ipynb')==pin['notebook_sha256']
    # SDK and numerical dependencies live in their existing isolated venvs.
    with (dest/'remote-source-inspection.log').open('w',encoding='utf8') as log:
        subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'scripts/inspect_c49_remote.py',trial],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    remote=json.loads((dest/'remote-source-audit.json').read_text())
    assert remote['private'] and not remote['internet'] and remote['version']==1 and remote['remote_source_exact']
    assert remote['source_sha256']==sha(ROOT/'scripts/inspect_c49_remote.py')
    remote_sha=remote['scientific_sha256'];assert remote_sha==pin['scientific_sha256']
    pattern=r'(c42_.*\.(csv|json|parquet|log)|c42_scores/.*\.(csv|parquet|json|npz)|execution_receipt\.json|p16_execution_receipt\.json|neural_(public|ours)\.parquet|neural_candidate_features\.parquet|effective_query_spectra\.parquet|submission\.csv)$'
    subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'scripts/fetch_output.py',slot,str(dest),pattern],cwd=ROOT,check=True,timeout=1800)
    r=json.loads((dest/'c42_execution.json').read_text())
    assert r['trial']==trial and not r['is_hidden_rerun'] and r['valid_rows']==400 and r['actual_query_count']==12
    assert not r['query_labels_mounted'] and not r['training_performed'] and not r['validation_reference_masks_mounted']
    assert r['pipeline']['processed_molecules']==12 and not r['pipeline']['is_full_run']
    assert r['exact_unprocessed_fallback_rows']==388
    assert len(r['devices'])==2 and all('T4' in d['name'] for d in r['devices'])
    head=ROOT/entry['bundle'];scorer='score_c48_crossmodal.py' if trial=='cpu' else 'score_c49_consensus.py'
    assert r['head_manifest_sha256']==pin['head_manifest_sha256']==sha(head/'head-manifest.json')
    assert sha(dest/'submission.csv')==r['csv_sha256'] and sha(dest/'c42_parent.csv')==r['parent_csv_sha256']
    p16=json.loads((dest/'p16_execution_receipt.json').read_text());assert p16['csv_sha256']==r['parent_csv_sha256']
    previous=ROOT/'outputs/forks/p20-c42-cpu-alignment-v1/c42_parent.csv'
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
        subprocess.run([str(ROOT/'.venv-gpu/Scripts/python.exe'),'scripts/audit_c48_ranking.py' if trial=='cpu' else 'scripts/audit_c49_ranking.py',str(scores),str(dest/'c42_active_parent.csv')],cwd=ROOT,check=True)
    local=dest/'local-head-replay'
    if not (local/'score-receipt.json').exists():
        assert not local.exists(),'Partial local replay; inspect it before recovery'
        with (dest/'local-head-replay.log').open('w') as log:
            subprocess.run([str(ROOT/'.venv-encoders/Scripts/python.exe'),str(head/scorer),str(head),
                str(ROOT/'outputs/publish/frozen-p14-20261107'),str(dest/'c42_active_parent.csv'),str(dest/'effective_query_spectra.parquet'),
                str(local),'--only',trial],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,check=True)
    cloud_score=pd.read_parquet(scores/'candidate-scores.parquet').set_index(['trial','molecule_id','parent_rank','smiles'])
    local_score=pd.read_parquet(local/'candidate-scores.parquet').set_index(cloud_score.index.names).reindex(cloud_score.index)
    assert cloud_score.eligible.equals(local_score.eligible)
    errors={column:float(np.abs(cloud_score[column]-local_score[column]).max()) for column in (['score'] if trial=='cpu' else ['score','reference_score'])};assert all(np.isfinite(v) and v<.005 for v in errors.values()),errors;error=max(errors.values())
    assert receipt['supported_queries']==11 and receipt['queries']==12 and not receipt['feature_cache_used']
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
        local_cloud_final_lists_exact=True,local_cloud_score_max_abs_difference=error,local_cloud_score_differences=errors,unchanged_P01_blend_max_abs_error=parent_error,
        parent_preview_exact_previous=True,quality_labels_read=False,C13_read=False,statistical_promotion=False,
        artifact_sha256={str(p.relative_to(ROOT)):sha(p) for p in [dest/'c42_execution.json',dest/'submission.csv',
            scores/'score-receipt.json',scores/'independent-ranking-audit.json',local/'score-receipt.json',folder/'production-pin.json',
            dest/'remote-source-audit.json',ROOT/'scripts/inspect_c49_remote.py']})
    write(target,result);print('C49 PRODUCTION AUDIT PASS',json.dumps(result),flush=True)


if __name__=='__main__':main()
