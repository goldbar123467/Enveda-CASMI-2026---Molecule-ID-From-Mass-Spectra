"""Independent exact full/incremental/mode C13arithmetic after one consumption."""
import numpy as np
import pandas as pd
from rdkit import Chem,DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from c45_complete_parent import ROOT,read,write,sha,stamp
from audit_c45_quality import check_point
OUT=ROOT/'outputs/local/c61-paired-confirmation-20261112'
BASE=ROOT/'outputs/local/c60-complete-c13-20261112'
C13=ROOT/'outputs/local/c13-confirmation-20261110'
MODELS=ROOT/'outputs/local/c56-bounded-semantic-heads-20261112'


def main():
    target=OUT/'independent-confirmation-audit.json';assert not target.exists()
    result=read(OUT/'confirmation-results.json');frozen=read(OUT/'prediction-freeze.json');marker=read(C13/'evaluation-consumption.json')
    assert marker['state']=='completed' and marker['results_sha256']==sha(OUT/'confirmation-results.json')
    assert result['commitment']['prediction_freeze_sha256']==sha(OUT/'prediction-freeze.json')
    assert result['commitment']['source_sha256']==frozen['source_sha256']==sha(ROOT/'scripts/confirm_c61_pair.py')
    assert all(marker.get(k)==v for k,v in result['commitment'].items())
    for name,pin in frozen['files_sha256'].items():assert sha(ROOT/name)==pin
    assert frozen['C13_labels_sha256']==sha(C13/'labels.csv')==read(C13/'preparation-report.json')['labels_sha256']
    parent=pd.read_csv(BASE/'P01.csv',dtype=str,keep_default_na=False);mids=parent.molecule_id.tolist();assert len(mids)==len(set(mids))==120
    labels=pd.read_csv(C13/'labels.csv',dtype=str).set_index('molecule_id').loc[mids]
    q=pd.read_parquet(BASE/'queries.parquet');modes=q.groupby('molecule_id').ionization_mode.agg(lambda a:'mixed' if a.nunique()>1 else a.iloc[0]).loc[mids].to_numpy()
    assert labels['mode'].tolist()==list(modes) and labels['mode'].value_counts().to_dict()=={'positive':60,'negative':60}
    frames=dict(P01=parent,P16=pd.read_csv(BASE/'P16.csv',dtype=str,keep_default_na=False))
    for trial in ['cpu','gpu']:frames[trial]=pd.read_csv(BASE/'head-predictions'/(trial+'.csv'),dtype=str,keep_default_na=False)
    identity=pd.read_parquet(OUT/'quality-identity-map.parquet').set_index('smiles').key.to_dict();ranks={};lists={};stored=np.load(OUT/'confirmation-ranks.npz')
    for name,frame in frames.items():
        assert frame.molecule_id.tolist()==mids;lists[name]=[[identity[s] for s in text.split(';')] for text in frame.smiles]
        ranks[name]=np.asarray([{key:j for j,key in enumerate(row,1)}.get(truth,0) for row,truth in zip(lists[name],labels.key)],np.int16)
        np.testing.assert_array_equal(ranks[name],stored[name])
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048);mols=[Chem.MolFromSmiles(s) for s in labels.smiles]
    assert all(m is not None for m in mols);fps=[gen.GetFingerprint(m) for m in mols];scaf=[MurckoScaffold.MurckoScaffoldSmiles(mol=m,includeChirality=False) for m in mols]
    parent_group=list(range(120))
    def root(i):
        if parent_group[i]!=i:parent_group[i]=root(parent_group[i])
        return parent_group[i]
    for i in range(120):
        for j in range(i):
            if labels.key.iloc[i]==labels.key.iloc[j] or labels.formula.iloc[i]==labels.formula.iloc[j] or bool(scaf[i] and scaf[i]==scaf[j]) or DataStructs.TanimotoSimilarity(fps[i],fps[j])>=.7:parent_group[root(i)]=root(j)
    groups=np.load(OUT/'confirmation-groups.npy')
    for i in range(120):
        for j in range(i):assert (root(i)==root(j))==(groups[i]==groups[j])
    fitting_ids=np.load(MODELS/'fitting-identities.npy');fit_rows=np.load(MODELS/'fitting-spectrum-rows.npy')
    cache=ROOT/'outputs/local/long-frozen-20261107-r1';mol=pd.read_parquet(cache/'molecules.parquet',columns=['key']);selection=pd.read_parquet(cache/'selected.parquet',columns=['row'])
    raw=pd.read_parquet(ROOT/'outputs/competition-raw/train.parquet',columns=['inchikey14']);raw_keys=set(raw.inchikey14.iloc[selection.row.to_numpy()[fit_rows]].astype(str))
    w=pd.read_parquet(MODELS/'original-structure-identity-witnesses.parquet',columns=['raw_key','official_key']);official=set(w.loc[w.raw_key.isin(raw_keys),'official_key'].dropna())|set(mol.key.iloc[fitting_ids])
    prior_keys=set()
    for path in ['outputs/local/c39-source-supported-validation-20261111/labels.csv','outputs/local/c48-auxiliary-holdout-20261112/labels.csv','outputs/local/c53-target-holdout-20261112/labels.csv']:prior_keys.update(pd.read_csv(ROOT/path,usecols=['key']).key)
    fitting_overlap=int(labels.key.isin(official).sum());prior_overlap=int(labels.key.isin(prior_keys).sum())
    assert result['fitting_exposure']['new_head_identity_overlaps']==fitting_overlap and result['fitting_exposure']['prior_used_identity_overlaps']==prior_overlap
    def check(v,base,trial):
        signs=[check_point(v['all120']['point'],ranks[base],ranks[trial],np.ones(120,bool))]
        assert set(v['mode'])=={'positive','negative'}
        for mode,entry in v['mode'].items():signs.append(check_point(entry['point'],ranks[base],ranks[trial],modes==mode))
        assert v['recovered_truths']==int(((ranks[base]==0)&(ranks[trial]>0)).sum())
        assert v['displaced_truths']==int(((ranks[base]>0)&(ranks[trial]==0)).sum())
        assert v['truth_absent_queries']==int((ranks[trial]==0).sum())
        assert v['rank1_changes']==sum(a[0]!=b[0] for a,b in zip(lists[base],lists[trial]))
        assert v['candidate_membership_changes']==sum(set(a)!=set(b) for a,b in zip(lists[base],lists[trial]))
        return all(s>=0 for s in signs)
    admissions={}
    for trial,entry in result['results'].items():
        full=check(entry['complete'],'P01',trial);incremental=check(entry['incremental'],'P16',trial)
        admissions[trial]=full and incremental and fitting_overlap==prior_overlap==0
        assert admissions[trial]==entry['admission']['passed']
        for a,b in zip(lists['P16'],lists[trial]):assert a[:2]==b[:2] and a[10:]==b[10:] and set(a)==set(b)
    check(result['parent_overlay'],'P01','P16');assert result['passed_pair']==all(admissions.values())
    assert result['one_joint_reserved_evaluation'] and result['all_absent_truths_zero'] and result['no_recipe_or_cohort_selection']
    write(target,dict(completed_utc=stamp(),source_sha256=sha(__file__),confirmation_results_sha256=sha(OUT/'confirmation-results.json'),
        prediction_freeze_sha256=sha(OUT/'prediction-freeze.json'),global_consumption_sha256=sha(C13/'evaluation-consumption.json'),
        queries=120,acquisitions=120,exact_official_rank_vectors=True,exact_fraction_full_incremental_and_mode_points=True,
        all120_and_both60mode_denominators_exact=True,all_absent_truths_zero=True,query_chemistry_groups_independently_exact=True,
        actual_fitted_and_original_row_identity_overlap=fitting_overlap,all_prior_used_identity_overlap=prior_overlap,
        recovered_displaced_counts_exact=True,admissions=admissions,passed_pair=all(admissions.values()),statistical_promotion=False,leaderboard_calibration=False))
    print('C61 INDEPENDENT COMPLETE CONFIRMATION AUDIT',admissions,flush=True)


if __name__=='__main__':main()
