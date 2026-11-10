"""Independent model, fusion, identity, fallback and local/cloud smoke audit."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
import argparse,hashlib,importlib.util,json
from pathlib import Path
from functools import lru_cache
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import torch
from rdkit import rdBase,RDLogger
from precision_cache import ROOT,sha,write,features
from submit_c09_external_once import scientific_sha

OUT=ROOT/'outputs/local/two-past-trials-20261110'
FORK=ROOT/'kaggle/forks/p18-p19-past-trained-trials'

def main():
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args();folder=a.folder
    assert not (folder/'independent_past_trials_audit.json').exists(),'Completed audit is sealed'
    assert rdBase.rdkitVersion=='2026.03.3';torch.set_num_threads(2);RDLogger.DisableLog('rdApp.*')
    freeze=json.loads((OUT/'production-freeze.json').read_text());r=json.loads((folder/'past_trials_execution.json').read_text())
    assert sha(FORK/'notebook.ipynb')==freeze['notebook_sha256']
    nb=json.loads((FORK/'notebook.ipynb').read_text(encoding='utf-8'));old=json.loads((ROOT/'kaggle/forks/p16-external-reference-c09/notebook.ipynb').read_text(encoding='utf-8'))
    assert nb['cells'][:-1]==old['cells'] and scientific_sha(nb)==freeze['scientific_sha256']
    assert sha(ROOT/'kaggle/forks/p16-external-reference-c09/notebook.ipynb')==freeze['parent_notebook_sha256']
    assert r['models']==freeze['models'] and r['selection_sha256']==freeze['selection_sha256']==sha(OUT/'selection.json')
    assert not r['is_hidden_rerun'] and r['pipeline']['processed_molecules']==12 and not r['pipeline']['is_full_run']
    assert len(r['devices'])==2 and all(d['name']=='Tesla T4' for d in r['devices'])
    parent=pd.read_csv(folder/'past_trials_parent.csv',dtype=str,keep_default_na=False)
    queries=pd.read_parquet(folder/'effective_query_spectra.parquet')
    p16=json.loads((folder/'p16_execution_receipt.json').read_text())
    assert sha(folder/'past_trials_parent.csv')==r['parent_csv_sha256']==p16['csv_sha256']
    assert len(parent)==400 and parent.molecule_id.is_unique and queries.molecule_id.nunique()==12
    @lru_cache(maxsize=30000)
    def identity(s):return features.structure(s)[0]
    all_reports={}
    for slot,entry in freeze['models'].items():
        bundle=ROOT/entry['local_bundle'];manifest=json.loads((bundle/'c08-manifest.json').read_text())
        assert sha(bundle/'c08-manifest.json')==entry['manifest_sha256']
        assert manifest['steps']==entry['steps'] and [m['fold'] for m in manifest['models']]==[0,1]
        assert not manifest['smoke']
        for name,pin in entry['files_sha256'].items():assert sha(bundle/name)==pin,name
        scores=pd.read_parquet(folder/(slot+'_scores.parquet'));answer=pd.read_csv(folder/entry['output'],dtype=str,keep_default_na=False)
        assert answer.columns.tolist()==['molecule_id','smiles'] and answer.molecule_id.tolist()==parent.molecule_id.tolist()
        assert sha(folder/entry['output'])==r['recipes'][slot]['csv_sha256']
        assert r['recipes'][slot]['input_errors']==0 and r['recipes'][slot]['scored_molecules']==12
        groups={str(mid):g.sort_values('baseline_rank',kind='stable') for mid,g in scores.groupby('molecule_id',sort=False)}
        errors=0.;changed=protected=0
        for before,after in zip(parent.itertuples(index=False),answer.itertuples(index=False)):
            original=before.smiles.split(';');actual=after.smiles.split(';');g=groups[str(before.molecule_id)]
            assert g.smiles.tolist()==original and len(actual)==len(original) and set(actual)==set(original) and actual[0]==original[0]
            keys=[identity(s) for s in actual];assert all(keys) and len(keys)==len(set(keys)) and 1<=len(keys)<=25
            if g.c08_ok.astype(bool).all():
                expected=np.zeros(len(g),np.float64)
                for fold in [0,1]:
                    raw=g['c08_raw_fold'+str(fold)].to_numpy(np.float64);expected+=.5*(raw-raw.mean())/max(float(raw.std()),1e-6)
                errors=max(errors,float(np.abs(expected-g.c08_score.to_numpy()).max()))
                neural_order=sorted(range(len(g)),key=lambda i:(-expected[i],i));ranks={i:rank+1 for rank,i in enumerate(neural_order)}
                utility={i:1/(4+i)+.2/(3+ranks[i]) for i in range(len(g))}
                order=[0]+sorted(range(1,len(g)),key=lambda i:(-utility[i],i))
                assert actual==[original[i] for i in order],'Independent fusion mismatch'
            else:assert actual==original;protected+=1
            changed+=int(actual!=original)
        assert changed>0 and protected==388 and errors<1e-10
        spec=importlib.util.spec_from_file_location('_past_'+slot,bundle/'inference.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        local_path=folder/(slot+'_local_scores.parquet')
        if local_path.exists():
            receipt=json.loads(Path(str(local_path)+'.json').read_text());assert receipt['manifest_sha256']==entry['manifest_sha256']
            assert receipt['queries_sha256']==sha(folder/'effective_query_spectra.parquet')
            assert receipt['baseline_content_sha256']==hashlib.sha256(parent.to_csv(index=False,lineterminator='\n').encode()).hexdigest()
            local=pd.read_parquet(local_path)
        else:local,receipt=module.score(bundle,entry['arm'],parent,folder/'effective_query_spectra.parquet',local_path)
        index=['molecule_id','smiles'];c=scores.set_index(index);l=local.set_index(index).reindex(c.index)
        assert not c.index.has_duplicates and l.c08_ok.equals(c.c08_ok)
        difference={name:float(np.abs(c[name].to_numpy()-l[name].to_numpy()).max()) for name in ['c08_raw_fold0','c08_raw_fold1','c08_score']}
        assert max(difference.values())<1e-3,('Local/cloud forward mismatch',slot,difference)
        from compatible_tail import apply
        local_result,_=apply(parent,local,.2);assert local_result.equals(answer),'Local/cloud final lists differ'
        all_reports[slot]=dict(valid_rows=400,real_queries=12,changed_rows=changed,exact_fallback_rows=protected,official_identity_duplicates=0,
            rank_one_changes=0,candidate_set_changes=0,independent_standardization_error=errors,independent_tail_arithmetic_exact=True,
            local_cloud_forward_max_abs_difference=difference,local_cloud_final_lists_exact=True,manifest_sha256=entry['manifest_sha256'],
            model_files_sha256={m['file']:sha(bundle/m['file']) for m in manifest['models']},csv_sha256=sha(folder/entry['output']),output=entry['output'])
    key=['molecule_id','smiles'];public=pd.read_parquet(folder/'neural_public.parquet').set_index(key)
    ours=pd.read_parquet(folder/'neural_ours.parquet').set_index(key).reindex(public.index)
    blended=pd.read_parquet(folder/'neural_candidate_features.parquet').set_index(key).reindex(public.index)
    def z(frame):
        s=frame.neural_score.fillna(0);g=s.groupby(level=0);return (s-g.transform('mean'))/g.transform(lambda x:x.std(ddof=0)).clip(lower=1e-6)
    error=float(np.abs(blended.neural_score-.5*z(public)-.5*z(ours)).max());assert error<1e-6
    report=dict(completed_utc=datetime.now(timezone.utc).isoformat(),execution_qualified=True,source_parent_exact=True,
        unchanged_P01_primary_error=error,independent_audit_source_sha256=sha(__file__),notebook_sha256=freeze['notebook_sha256'],
        remote_scientific_sha256=freeze['scientific_sha256'],selection_sha256=freeze['selection_sha256'],recipes=all_reports,
        training_performed=False,reserved_labels_read=False,statistical_promotion=False,explicit_user_trials=2,maximum_pending=1,
        limitation='Already-used source-shifted development; both chosen confidence intervals include zero. Scoring trials do not establish a hidden gain.')
    write(folder/'independent_past_trials_audit.json',report);print('DONE INDEPENDENT TWO-TRIAL AUDIT',json.dumps(report),flush=True)

if __name__=='__main__':main()
