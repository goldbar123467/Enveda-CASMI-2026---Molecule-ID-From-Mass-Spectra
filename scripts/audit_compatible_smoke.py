"""Independent P15 artifact, arithmetic, identity and local/Kaggle score audit."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from functools import lru_cache
import argparse,json
from rdkit import Chem,RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
from compatible_inference import score,digest
from precision_cache import ROOT,write


def main():
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path);args=parser.parse_args();folder=args.folder
    torch.set_num_threads(2);RDLogger.DisableLog('rdApp.*')
    bundle=ROOT/'outputs/publish/c08-models-20261108';manifest=json.loads((bundle/'c08-manifest.json').read_text())
    pin=json.loads((ROOT/'kaggle/forks/p15-compatible-c08/model-pin.json').read_text())
    receipt=json.loads((folder/'p15_qualification_receipt.json').read_text())
    assert not manifest['smoke'] and manifest['steps']==40000 and len(manifest['models'])==8
    assert receipt['manifest_sha256']==pin['manifest_sha256']==digest(bundle/'c08-manifest.json')
    assert receipt['model']['selection_sha256']==pin['selection_sha256']==digest(bundle/'selection.json')
    assert receipt['arm']==pin['arm'] and receipt['weight']==pin['weight'] and not receipt['smoke']
    assert receipt['models']==2 and receipt['folds']==[0,1] and receipt['steps']==40000 and receipt['input_errors']==0
    assert receipt['scored_molecules']==12 and receipt['rank_one_changes']==receipt['candidate_set_changes']==0
    assert receipt['merge_status']=='applied' and not receipt['is_hidden_rerun']
    for name in ['compatible_inference.py','compatible_tail.py']:assert digest(ROOT/'scripts'/name)==manifest['files'][name]
    before=pd.read_csv(folder/'c08_p01_submission.csv',dtype=str,keep_default_na=False)
    after=pd.read_csv(folder/'submission.csv',dtype=str,keep_default_na=False)
    assert after.columns.tolist()==before.columns.tolist()==['molecule_id','smiles'] and len(after)==400
    assert after.molecule_id.is_unique and after.molecule_id.tolist()==before.molecule_id.tolist()
    scores=pd.read_parquet(folder/'c08_candidate_scores.parquet');groups={mid:g.sort_values('baseline_rank',kind='stable') for mid,g in scores.groupby('molecule_id',sort=False)}
    taut=rdMolStandardize.TautomerEnumerator()
    @lru_cache(maxsize=20000)
    def identity(s):
        mol=Chem.MolFromSmiles(s);assert mol is not None;Chem.RemoveStereochemistry(mol)
        return Chem.MolToInchiKey(taut.Canonicalize(mol))[:14]
    changed=0;blend_error=0.
    for old,new in zip(before.itertuples(index=False),after.itertuples(index=False)):
        values=old.smiles.split(';');actual=new.smiles.split(';');g=groups[old.molecule_id]
        assert g.smiles.tolist()==values
        assert 1<=len(actual)<=25 and actual[0]==values[0] and len(actual)==len(values) and set(actual)==set(values)
        keys=[identity(s) for s in actual];assert len(set(keys))==len(keys) and all(len(k)==14 for k in keys)
        if g.c08_ok.all():
            expected=np.zeros(len(g),np.float64)
            for fold in [0,1]:
                raw=g[f'c08_raw_fold{fold}'].to_numpy(dtype=np.float64);expected+=.5*(raw-raw.mean())/max(float(raw.std()),1e-6)
            blend_error=max(blend_error,float(np.abs(expected-g.c08_score.to_numpy()).max()))
            ordering=sorted(range(len(expected)),key=lambda i:(-expected[i],i));neural={i:rank+1 for rank,i in enumerate(ordering)}
            total={i:1/(3+i+1)+pin['weight']/(3+neural[i]) for i in range(len(expected))}
            indices=[0]+sorted(range(1,len(expected)),key=lambda i:(-total[i],i))
            assert actual==[values[i] for i in indices]
        else:assert actual==values
        changed+=int(actual!=values)
    assert blend_error<1e-6
    local_path=folder/'c08_local_scores.parquet'
    local,_=score(bundle,pin['arm'],before,folder/'effective_query_spectra.parquet',local_path)
    key=['molecule_id','smiles'];a=scores.set_index(key);b=local.set_index(key).reindex(a.index)
    assert not a.index.has_duplicates and b.c08_ok.equals(a.c08_ok)
    differences={c:float(np.abs(a[c].to_numpy()-b[c].to_numpy()).max()) for c in ['c08_raw_fold0','c08_raw_fold1','c08_score']}
    assert max(differences.values())<1e-3,('Local/Kaggle raw-score mismatch',differences)
    # Verify the primary P01 blend remains exact before our tail-only stage.
    public=pd.read_parquet(folder/'neural_public.parquet').set_index(key)
    v80=pd.read_parquet(folder/'neural_ours.parquet').set_index(key).reindex(public.index)
    final=pd.read_parquet(folder/'neural_candidate_features.parquet').set_index(key).reindex(public.index)
    def z(frame):
        s=frame.neural_score.fillna(0);g=s.groupby(level=0)
        return (s-g.transform('mean'))/g.transform(lambda v:v.std(ddof=0)).clip(lower=1e-6)
    p01_error=float(np.abs(final.neural_score-.5*z(public)-.5*z(v80)).max());assert p01_error<1e-6
    report=dict(valid_rows=400,real_molecules=12,input_errors=0,official_identity_duplicates=0,rank_one_changes=0,candidate_set_changes=0,
        changed_rows=changed,independent_twofit_standardization_error=blend_error,independent_tail_arithmetic=True,
        unchanged_p01_primary_score_error=p01_error,local_Kaggle_score_max_abs_difference=differences,
        model=receipt['model'],csv_sha256=digest(folder/'submission.csv'),smoke_only=True,statistical_promotion=False,explicit_user_trial=True)
    write(folder/'independent_c08_smoke_audit.json',report);print('DONE INDEPENDENT P15 SMOKE AUDIT',json.dumps(report),flush=True)


if __name__=='__main__':main()
