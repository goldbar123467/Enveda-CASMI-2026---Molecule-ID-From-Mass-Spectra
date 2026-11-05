"""Independent numeric, schema and official-identity checks of a real pipeline smoke."""
import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem,RDLogger,__version__ as rdkit_version
from rdkit.Chem.MolStandardize import rdMolStandardize


def main():
    ap=argparse.ArgumentParser();ap.add_argument('folder',type=Path)
    args=ap.parse_args();folder=args.folder
    receipts=list(folder.glob('*qualification_receipt.json'));assert len(receipts)==1
    receipt=json.loads(receipts[0].read_text())
    assert receipt['real_scored_molecules']==12 and receipt['merge_status']=='applied'
    assert not receipt['is_hidden_rerun'] and not receipt['pipeline']['is_full_run']
    assert receipt['neural_score_max_abs_error']<1e-6
    sub=pd.read_csv(folder/'submission.csv',dtype=str,keep_default_na=False)
    base=pd.read_csv(folder/'submission_baseline0465.csv',dtype=str,keep_default_na=False)
    assert sub.columns.tolist()==base.columns.tolist()==['molecule_id','smiles']
    assert len(sub)==400 and sub.molecule_id.is_unique and sub.molecule_id.tolist()==base.molecule_id.tolist()
    RDLogger.DisableLog('rdApp.*');tautomer=rdMolStandardize.TautomerEnumerator()
    @lru_cache(maxsize=100000)
    def identity(smiles):
        molecule=Chem.MolFromSmiles(smiles);assert molecule is not None
        Chem.RemoveStereochemistry(molecule)
        key=Chem.MolToInchiKey(tautomer.Canonicalize(molecule))[:14]
        assert len(key)==14
        return key
    for row in sub.itertuples(index=False):
        guesses=row.smiles.split(';');assert 1<=len(guesses)<=25
        keys=[identity(s) for s in guesses];assert len(keys)==len(set(keys)),row.molecule_id
    public=pd.read_parquet(folder/'neural_public.parquet').set_index(['molecule_id','smiles'])
    ours=pd.read_parquet(folder/'neural_ours.parquet').set_index(['molecule_id','smiles']).reindex(public.index)
    actual=pd.read_parquet(folder/'neural_candidate_features.parquet').set_index(['molecule_id','smiles']).reindex(public.index)
    eligible=public.neural_ok.astype(bool)
    assert ours.loc[eligible,'neural_ok'].astype(bool).all()
    difference=float((public.loc[eligible,'neural_score']-ours.loc[eligible,'neural_score']).abs().max())
    assert difference>1e-6,'Second neural scorer did not change any real candidate score'
    def z(values):
        groups=values.groupby(level=0)
        return (values-groups.transform('mean'))/groups.transform(lambda s:s.std(ddof=0)).clip(lower=1e-6)
    expected=.5*z(public.neural_score.fillna(0))+.5*z(ours.neural_score.fillna(0))
    error=float((actual.neural_score-expected).abs().max())
    assert np.isfinite(actual.loc[eligible,'neural_score']).all() and error<1e-4
    report=dict(valid_rows=len(sub),real_molecules=receipt['real_scored_molecules'],
                official_identity_duplicates=0,rdkit=rdkit_version,canonicalized_unique_smiles=identity.cache_info().currsize,
                independent_blend_max_abs_error=error,second_model_score_max_difference=difference,
                changed_rows_vs_frozen=int(sub.smiles.ne(base.smiles).sum()),
                csv_sha256=hashlib.sha256((folder/'submission.csv').read_bytes()).hexdigest(),
                model=receipt['model'],smoke_only=True,competition_score=None)
    (folder/'independent_smoke_audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='model'},indent=2))


if __name__=='__main__':
    main()
