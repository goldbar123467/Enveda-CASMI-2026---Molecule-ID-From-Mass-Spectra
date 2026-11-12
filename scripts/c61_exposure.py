"""Source-pinned exposure/grouping report after the single C13consumption."""
import numpy as np
import pandas as pd
from rdkit import Chem,DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from c45_complete_parent import ROOT,read,sha
OUT=ROOT/'outputs/local/c61-paired-confirmation-20261112'
C40=ROOT/'outputs/local/c56-bounded-semantic-heads-20261112'
CACHE=ROOT/'outputs/local/long-frozen-20261107-r1'
def authorize():
    a=read(ROOT/'.local/c38-active-goal.json');assert a['active'] and a['submissions_approved']==2 and a['submissions_used']==0

def chemistry_and_fit_exposure(labels):
    """Report every query's neighbor exposure; never filter a favorable cohort."""
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
    querymol=[Chem.MolFromSmiles(s) for s in labels.smiles];assert all(m is not None for m in querymol)
    fp=[gen.GetFingerprint(m) for m in querymol]
    scaffold=[MurckoScaffold.MurckoScaffoldSmiles(mol=m,includeChirality=False) for m in querymol]
    roots=list(range(len(labels)))
    def find(i):
        while roots[i]!=i:roots[i]=roots[roots[i]];i=roots[i]
        return i
    for i in range(len(labels)):
        for j in range(i):
            if labels.key.iloc[i]==labels.key.iloc[j] or labels.formula.iloc[i]==labels.formula.iloc[j] or (scaffold[i] and scaffold[i]==scaffold[j]) or DataStructs.TanimotoSimilarity(fp[i],fp[j])>=.7:roots[find(i)]=find(j)
    groups=np.asarray([find(i) for i in range(len(labels))])
    ids=np.load(C40/'fitting-identities.npy',allow_pickle=False);rows=np.load(C40/'fitting-spectrum-rows.npy',allow_pickle=False)
    boundary=json.loads((C40/'fit-boundary-freeze.json').read_text())
    for n in ['fitting-identities.npy','fitting-spectrum-rows.npy']:assert sha(C40/n)==boundary['files_sha256'][n]
    mol=pd.read_parquet(CACHE/'molecules.parquet').iloc[ids]
    selected=pd.read_parquet(CACHE/'selected.parquet',columns=['row']);actual_rows=selected.row.to_numpy()[rows]
    raw=pd.read_parquet(ROOT/'outputs/competition-raw/train.parquet',columns=['inchikey14'])
    actual_raw_keys=set(raw.inchikey14.iloc[actual_rows].astype(str));del raw,selected
    witnesses=pd.read_parquet(ROOT/'outputs/local/c56-bounded-semantic-heads-20261112/original-structure-identity-witnesses.parquet',columns=['raw_key','official_key'])
    fitted_original_official=set(witnesses.loc[witnesses.raw_key.isin(actual_raw_keys),'official_key'].dropna())
    overlaps=labels.key.isin(set(mol.key)|fitted_original_official).to_numpy()
    prior_labels=pd.concat([pd.read_csv(ROOT/p) for p in ['outputs/local/c39-source-supported-validation-20261111/labels.csv','outputs/local/c48-auxiliary-holdout-20261112/labels.csv','outputs/local/c53-target-holdout-20261112/labels.csv']],ignore_index=True).drop_duplicates('key')
    prior_identity=labels.key.isin(set(prior_labels.key)).to_numpy();prior_maximum=np.zeros(len(labels))
    prior_formula=np.zeros(len(labels),np.int64);prior_scaffold=np.zeros(len(labels),np.int64)
    for row in prior_labels.itertuples(index=False):
        m=Chem.MolFromSmiles(row.smiles);assert m is not None
        prior_maximum=np.maximum(prior_maximum,DataStructs.BulkTanimotoSimilarity(gen.GetFingerprint(m),fp))
        prior_formula+=np.asarray([row.formula==v for v in labels.formula])
        sc=MurckoScaffold.MurckoScaffoldSmiles(mol=m,includeChirality=False)
        if sc:prior_scaffold+=np.asarray([sc==v for v in scaffold])
    maximum=np.zeros(len(labels));sameformula=np.zeros(len(labels),np.int64);samescaffold=np.zeros(len(labels),np.int64)
    for count,row in enumerate(mol.itertuples(index=False),1):
        m=Chem.MolFromSmiles(row.smiles);assert m is not None
        maximum=np.maximum(maximum,DataStructs.BulkTanimotoSimilarity(gen.GetFingerprint(m),fp))
        sameformula+=np.asarray([row.formula==v for v in labels.formula])
        sc=MurckoScaffold.MurckoScaffoldSmiles(mol=m,includeChirality=False)
        if sc:samescaffold+=np.asarray([sc==v for v in scaffold])
        if count%25000==0:authorize();print('C61 FIT EXPOSURE CHECK',count,'/',len(mol),flush=True)
    exposure=pd.DataFrame(dict(molecule_id=labels.index,fit_identity_overlap=overlaps,maximum_fit_tanimoto=maximum,
        fitting_same_formula=sameformula,fitting_same_nonempty_scaffold=samescaffold,prior_used_identity_overlap=prior_identity,
        prior_used_maximum_tanimoto=prior_maximum,prior_used_same_formula=prior_formula,prior_used_same_nonempty_scaffold=prior_scaffold,group=groups))
    exposure.to_csv(OUT/'fitting-exposure.csv',index=False)
    return groups,dict(new_head_identity_overlaps=int(overlaps.sum()),queries_with_fitting_tanimoto_at_least_point7=int((maximum>=.7).sum()),
        prior_used_identity_overlaps=int(prior_identity.sum()),queries_with_prior_used_tanimoto_at_least_point7=int((prior_maximum>=.7).sum()),
        queries_with_prior_used_same_formula=int((prior_formula>0).sum()),queries_with_prior_used_same_nonempty_scaffold=int((prior_scaffold>0).sum()),
        queries_with_fitting_same_formula=int((sameformula>0).sum()),queries_with_fitting_same_nonempty_scaffold=int((samescaffold>0).sum()),
        original_raw_identity_groups_checked=len(actual_raw_keys),query_groups=len(set(groups)),
        near_chemistry_excluded_from_all_training=False,universal_pretraining_exclusion=False)
