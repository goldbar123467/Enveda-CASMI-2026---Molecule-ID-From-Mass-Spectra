"""Independent fold/statistics, convex optimum, list and exact metric checks."""
import os
for n in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(n,'2')
import torch
import json
import numpy as np
import pandas as pd
from fractions import Fraction
from rdkit import Chem,DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from c45_complete_parent import ROOT,read,write,sha,stamp,module
from prepare_c64_pairwise_data import PANELS
DATA=ROOT/'outputs/local/c64-pairwise-oof-20261112'


def main():
    q=pd.read_parquet(DATA/'queries.parquet');rows=pd.read_parquet(DATA/'candidates.parquet');x=np.load(DATA/'features.npy').astype(np.float64)
    assert len(q)==869 and q.acquisitions.sum()==3452
    for cohort,n,_,label_file,*_ in PANELS:
        lab=pd.read_csv(ROOT/label_file,dtype=str).set_index('molecule_id');subset=q[q.cohort.eq(cohort)]
        assert len(subset)==n and subset.key.tolist()==lab.loc[subset.molecule_id,'key'].tolist()
    assert np.array_equal(rows.target,rows.key.eq(rows.molecule_id.map(q.set_index('molecule_id').key)).astype(int))
    names=read(DATA/'feature-names.json');assert not any(t in n for n in names for t in ['truth','target','source','key','molecule_id'])
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048);mols=[Chem.MolFromSmiles(s) for s in q.smiles]
    fps=[gen.GetFingerprint(m) for m in mols];sc=[MurckoScaffold.MurckoScaffoldSmiles(mol=m,includeChirality=False) for m in mols]
    for i in range(len(q)):
        sim=DataStructs.BulkTanimotoSimilarity(fps[i],fps[:i])
        for j in range(i):
            linked=q.key.iloc[i]==q.key.iloc[j] or q.formula.iloc[i]==q.formula.iloc[j] or bool(sc[i] and sc[i]==sc[j]) or sim[j]>=.7
            if linked:assert q.fold.iloc[i]==q.fold.iloc[j],'Chemistry edge crosses a held-out fold'
    pairs=[]
    for _,g in rows.groupby(['molecule_id','formula'],sort=False):
        if g.available.all() and len(g)>1 and g.target.sum()==1:
            p=int(g.index[g.target.eq(1)][0]);negative=g.index[g.target.eq(0)]
            for n in negative:pairs.append((p,int(n),1/len(negative)))
    a=np.array([v[0] for v in pairs]);b=np.array([v[1] for v in pairs]);plain=np.array([v[2] for v in pairs]);parent=-np.log(rows.parent_rank.to_numpy(float))
    identity=dict(zip(rows.smiles,rows.key));formula=dict(zip(rows.key,rows.formula));official=module('oof_audit_identity','c09e_features.py').structure
    old=pd.read_csv(ROOT/'outputs/local/c39-source-supported-validation-20261111/labels.csv',dtype=str).set_index('molecule_id')
    for tag,folder,policy_file,script in [
        ('C64','c64-pairwise-oof-20261112','c64-physics-pairwise-oof-20261112.json','train_c64_cpu_oof.py'),
        ('C65','c65-rrweighted-oof-20261112','c65-rrweighted-oof-20261112.json','train_c65_rrweighted_oof.py')]:
        out=ROOT/'outputs/local'/folder;target=out/'independent-oof-audit.json';assert not target.exists()
        result=read(out/'cpu-oof-results.json');policy=read(ROOT/'configs'/policy_file)
        assert result['source_sha256']==sha(ROOT/'scripts'/script) and result['policy_sha256']==sha(ROOT/'configs'/policy_file)
        assert result['data_freeze_sha256']==sha(DATA/'data-freeze.json')
        predictions=np.empty(len(rows));gradients=[]
        weights=plain.copy()
        if tag=='C65':weights*=np.abs(1/rows.parent_rank.to_numpy(float)[a]-1/rows.parent_rank.to_numpy(float)[b])
        for fold in range(5):
            train=rows.fold.ne(fold).to_numpy();held=~train;selected=train[a]&train[b]
            model=np.load(out/f'cpu-fold{fold}.npz');mean=x[train].mean(0);std=np.maximum(x[train].std(0),1e-4)
            np.testing.assert_allclose(mean,model['mean'],rtol=0,atol=1e-12);np.testing.assert_allclose(std,model['std'],rtol=0,atol=1e-12)
            z=np.clip((x-mean)/std,-10,10);w=model['weights'];delta=z[a[selected]]-z[b[selected]];offset=parent[a[selected]]-parent[b[selected]]
            norm=weights[selected]/weights[selected].sum();margin=offset+delta@w;sigmoid=1/(1+np.exp(-np.clip(margin,-80,80)))
            gradient=delta.T@((sigmoid-1)*norm)+2*policy['cpu']['ridge']*w
            residual=float(np.abs(gradient).max());assert residual<2e-5,('Not the unique regularized training-only optimum',tag,fold,residual)
            gradients.append(residual);predictions[held]=parent[held]+z[held]@w
            assert result['folds'][fold]['model_sha256']==sha(out/f'cpu-fold{fold}.npz')
        scores=pd.read_parquet(out/'cpu-oof-scores.parquet');np.testing.assert_allclose(scores.oof_score,predictions,rtol=0,atol=1e-10)
        lists={mid:g.sort_values('parent_rank').smiles.tolist() for mid,g in rows.groupby('molecule_id',sort=False)}
        for (mid,_),g in rows.groupby(['molecule_id','formula'],sort=False):
            if len(g)>1 and g.available.all():
                order=sorted(g.index,key=lambda i:(-predictions[i],int(rows.parent_rank.iloc[i])))
                for rank,text in zip(g.parent_rank,rows.smiles.iloc[order]):lists[mid][int(rank)-1]=text
        actual=pd.read_csv(out/'cpu-oof.csv',dtype=str);assert actual.molecule_id.tolist()==q.molecule_id.tolist()
        assert actual.smiles.tolist()==[';'.join(lists[mid]) for mid in q.molecule_id]
        stored=np.load(out/'cpu-oof-ranks.npz');ranks={n:[] for n in ['P01','P16','physical_parent','cpu']}
        for name in ranks:
            for item in q.itertuples(index=False):
                strings=lists[item.molecule_id] if name=='cpu' else getattr(item,name).split(';')
                for s in strings:
                    if s not in identity:identity[s]=official(s)[0]
                keys=[identity[s] for s in strings];ranks[name].append(keys.index(item.key)+1 if item.key in keys else 0)
            np.testing.assert_array_equal(ranks[name],stored[name])
        for baseline,comparisons in result['comparisons'].items():
            for name,entry in comparisons.items():
                if name=='all869':mask=np.ones(869,bool)
                elif name=='original192':mask=q.molecule_id.isin(old.index[old.partition.eq('development')]).to_numpy()
                elif name=='original352':mask=q.cohort.eq('C39').to_numpy()
                elif '/' not in name:mask=q.cohort.eq(name).to_numpy()
                else:
                    cohort,field,value=name.split('/',2);mask=(q.cohort.eq(cohort)&q[field].eq(value)).to_numpy()
                p=[Fraction(1,int(r)) if r else Fraction(0) for r in stored[baseline][mask]];c=[Fraction(1,int(r)) if r else Fraction(0) for r in stored['cpu'][mask]]
                delta=sum((v-u for u,v in zip(p,c)),Fraction(0))/len(p)
                assert entry['n']==len(p) and entry['numerator']==delta.numerator and entry['denominator']==delta.denominator
                assert entry['nonnegative']==(delta>=0) and entry['strictly_positive']==(delta>0)
        write(target,dict(completed_utc=stamp(),source_sha256=sha(__file__),result_sha256=sha(out/'cpu-oof-results.json'),queries=869,acquisitions=3452,
            query_chemistry_edges_never_cross_folds=True,train_only_scaling_exact=True,convex_training_objective_gradient_maxima=gradients,
            actual_saved_OOF_scores_and_lists_exact=True,all_original_source_mode_fraction_points_exact=True,absent_truths_zero=True,
            CPU_gate_passed=result['CPU_gate_before_GPU_passed'],C13_read=False,scoring_entries=0,prospective_qualification=False))
        print(tag,'INDEPENDENT OOF AUDIT PASS; gate',result['CPU_gate_before_GPU_passed'],'gradientmax',max(gradients),flush=True)


if __name__=='__main__':main()
