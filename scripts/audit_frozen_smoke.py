"""Independent P14 final-list fusion, fallback, byte-pin and metric CSV audit."""
import argparse,hashlib,json
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem,RDLogger,__version__ as rdkit_version
from rdkit.Chem.MolStandardize import rdMolStandardize


def z(value):
    groups=value.groupby(level=0)
    return (value-groups.transform('mean'))/groups.transform(lambda v:v.std(ddof=0)).clip(lower=1e-6)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path);folder=parser.parse_args().folder
    receipt=json.loads((folder/'p14_qualification_receipt.json').read_text());assert receipt['real_scored_molecules']==12
    assert receipt['scored_molecules']>0 and receipt['input_errors']==0 and receipt['merge_status']=='applied'
    assert receipt['preprocess_sort']=='stable_intensity_original_index'
    assert not receipt['is_hidden_rerun'] and not receipt['pipeline']['is_full_run']
    public,private,base,final=[pd.read_parquet(folder/name) for name in
        ['neural_public.parquet','neural_ours.parquet','neural_p01.parquet','neural_candidate_features.parquet']]
    columns=['molecule_id','smiles'];assert all(base[columns].equals(f[columns]) for f in [public,private,final])
    a=public.set_index(columns);b=private.set_index(columns)
    expected=.5*z(a.neural_score.fillna(0))+.5*z(b.neural_score.fillna(0))
    base_error=float(np.abs(base.neural_score.to_numpy()-expected.to_numpy()).max());assert base_error<1e-6
    weight=receipt['weight'];error=0.;fallback=0;unprocessed=0;used=0
    for mid,g in final.groupby('molecule_id',sort=False):
        old=base.loc[g.index,'neural_score'].to_numpy(dtype=float)
        if g.frozen_ok.astype(bool).all():
            v=g.frozen_score.to_numpy(dtype=float);standard=(v-v.mean())/max(v.std(),1e-6)
            predicted=(1-weight)*old+weight*(standard*max(old.std(),1e-6)+old.mean());used+=1
        else:
            predicted=old
            if g.processed.astype(bool).all():fallback+=1
            else:unprocessed+=1
        error=max(error,float(np.abs(g.neural_score.to_numpy()-predicted).max()))
    assert error<1e-6 and used==receipt['scored_molecules'] and fallback>=1
    for column in ['neural_cos','neural_fp','neural_std','neural_ok']:
        assert np.array_equal(base[column],final[column]),column
    pin=json.loads((Path(__file__).resolve().parents[1]/'kaggle/forks/p14-frozen-channel/model-pin.json').read_text())
    assert receipt['manifest_sha256']==pin['manifest_sha256']
    assert receipt['model']['v80_sha256']==pin['v80_sha256']
    assert receipt['head_sha256']=='81d117d0e81698ae46a4f3b0077c329c8710aae555bdba6331955c4fbe896350'
    submission=pd.read_csv(folder/'submission.csv',dtype=str,keep_default_na=False)
    frozen=pd.read_csv(folder/'submission_baseline0465.csv',dtype=str,keep_default_na=False)
    assert submission.columns.tolist()==['molecule_id','smiles'] and len(submission)==400 and submission.molecule_id.is_unique
    assert submission.molecule_id.tolist()==frozen.molecule_id.tolist()
    RDLogger.DisableLog('rdApp.*');enumerator=rdMolStandardize.TautomerEnumerator()
    @lru_cache(maxsize=100000)
    def identity(smiles):
        mol=Chem.MolFromSmiles(smiles);assert mol is not None
        Chem.RemoveStereochemistry(mol);return Chem.MolToInchiKey(enumerator.Canonicalize(mol))[:14]
    for row in submission.itertuples(index=False):
        values=row.smiles.split(';');assert 1<=len(values)<=25
        keys=[identity(s) for s in values];assert all(len(k)==14 for k in keys) and len(keys)==len(set(keys)),row.molecule_id
    report=dict(valid_rows=400,real_molecules=12,frozen_scored_molecules=used,exact_baseline_fallback_molecules=fallback,unprocessed_passthrough_molecules=unprocessed,
                preprocess_sort=receipt['preprocess_sort'],numpy_version=receipt['numpy_version'],
                input_errors=0,official_identity_duplicates=0,rdkit=rdkit_version,primary_baseline_max_abs_error=base_error,
                independent_frozen_fusion_max_abs_error=error,model=receipt['model'],
                csv_sha256=hashlib.sha256((folder/'submission.csv').read_bytes()).hexdigest(),smoke_only=True,competition_score=None)
    (folder/'independent_frozen_smoke_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
