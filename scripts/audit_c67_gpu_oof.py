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
DATA=ROOT/'outputs/local/c66-parent-evidence-oof-20261112'


def main():
    freeze=read(DATA/'data-freeze.json')
    for name,pin in freeze['files_sha256'].items():assert sha(ROOT/name)==pin,name
    original=ROOT/'outputs/local/c64-pairwise-oof-20261112'
    for name in ['queries.parquet','candidates.parquet','physical-parent-changes.json']:
        assert sha(DATA/name)==sha(original/name),name
    q=pd.read_parquet(DATA/'queries.parquet');rows=pd.read_parquet(DATA/'candidates.parquet');x=np.load(DATA/'features.npy').astype(np.float64)
    old_x=np.load(original/'features.npy');np.testing.assert_array_equal(x[:,:old_x.shape[1]],old_x)
    pop=pd.read_csv(ROOT/'outputs/research/pop-prior/pool_popularity.csv',usecols=['inchikey14','pop','np_coconut','np_lotus','src_natural_product'])
    pop=pop.drop_duplicates('inchikey14',keep='last').set_index('inchikey14')
    popularity=rows.key.map(pop['pop']).fillna(0).to_numpy(float)
    natural=rows.key.map(pop[['np_coconut','np_lotus','src_natural_product']].any(axis=1).astype(int)).fillna(0).to_numpy(float)
    frames=[]
    for panel in PANELS:
        raw=pd.read_parquet(ROOT/panel[4]/'c41_neural_p01.parquet')
        raw['nr']=raw.groupby('molecule_id',sort=False).neural_score.transform(lambda v:np.argsort(np.argsort(-v.to_numpy(),kind='stable'),kind='stable')+1)
        frames.append(raw)
    lookup=pd.concat(frames).set_index(['molecule_id','metric_key'])
    ix=pd.MultiIndex.from_arrays([rows.molecule_id,rows.key]);br=lookup.loc[ix,'baseline_rank'].to_numpy(float);nr=lookup.loc[ix,'nr'].to_numpy(float)
    added=[br,nr,1/(3+br),.75/(3+nr),popularity/20,natural,1/(3+br)+.75/(3+nr)+.005*popularity+.02*natural]
    for i,v in enumerate(added):
        np.testing.assert_array_equal(x[:,old_x.shape[1]+2*i],v.astype(np.float32))
        z=np.empty(len(v))
        for _,g in rows.groupby('molecule_id',sort=False):
            a=v[g.index];z[g.index]=np.clip((a-a.mean())/max(float(a.std()),1e-6),-10,10)
        np.testing.assert_allclose(x[:,old_x.shape[1]+2*i+1],z.astype(np.float32),rtol=0,atol=1e-6)
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
        ('C67','c67-gpu-parent-evidence-oof-20261112','c67-gpu-parent-evidence-oof-20261112.json','train_c67_gpu_oof.py')]:
        out=ROOT/'outputs/local'/folder;target=out/'independent-oof-audit.json';assert not target.exists()
        result=read(out/'gpu-oof-results.json');policy=read(ROOT/'configs'/policy_file)
        assert result['source_sha256']==sha(ROOT/'scripts'/script) and result['policy_sha256']==sha(ROOT/'configs'/policy_file)
        assert result['data_freeze_sha256']==sha(DATA/'data-freeze.json')
        predictions=np.empty(len(rows));gradients=[]
        weights=plain.copy()
        if tag=='C65':weights*=np.abs(1/rows.parent_rank.to_numpy(float)[a]-1/rows.parent_rank.to_numpy(float)[b])
        for fold in range(5):
            train=rows.fold.ne(fold).to_numpy();held=~train;selected=train[a]&train[b]
            model=np.load(out/f'gpu-fold{fold}.npz');mean=x[train].mean(0);std=np.maximum(x[train].std(0),1e-4)
            np.testing.assert_allclose(mean,model['mean'],rtol=0,atol=1e-12);np.testing.assert_allclose(std,model['std'],rtol=0,atol=1e-12)
            z=np.clip((x-mean)/std,-10,10);w=model['weights'];delta=z[a[selected]]-z[b[selected]];offset=parent[a[selected]]-parent[b[selected]]
            norm=weights[selected]/weights[selected].sum();margin=offset+delta@w;sigmoid=1/(1+np.exp(-np.clip(margin,-80,80)))
            gradient=delta.T@((sigmoid-1)*norm)+2*policy['cpu']['ridge']*w
            residual=float(np.abs(gradient).max());assert residual<2e-5,('Not the unique regularized training-only optimum',tag,fold,residual)
            gradients.append(residual)
            assert np.array_equal(w,np.load(DATA/f'cpu-fold{fold}.npz')['weights'])
            anchor=parent+z@w
            torch.manual_seed(policy['seed']+fold);torch.cuda.manual_seed_all(policy['seed']+fold)
            net=torch.nn.Sequential(torch.nn.Linear(x.shape[1],32),torch.nn.GELU(approximate='tanh'),torch.nn.Linear(32,1)).to('cuda:0')
            torch.nn.init.zeros_(net[2].weight);torch.nn.init.zeros_(net[2].bias)
            zz=torch.tensor(z,dtype=torch.float32,device='cuda:0');aa=torch.tensor(anchor,dtype=torch.float32,device='cuda:0')
            pa=torch.tensor(a[selected],device='cuda:0');pb=torch.tensor(b[selected],device='cuda:0')
            pw=torch.tensor(plain[selected],dtype=torch.float32,device='cuda:0');pw/=pw.sum()
            ix=torch.tensor(np.flatnonzero(train),device='cuda:0')
            opt=torch.optim.AdamW(net.parameters(),lr=policy['gpu']['learning_rate'],weight_decay=policy['gpu']['weight_decay'])
            for epoch in range(policy['gpu']['epochs']):
                opt.zero_grad();correction=.25*torch.tanh(net(zz).flatten());margin=aa[pa]-aa[pb]+correction[pa]-correction[pb]
                loss=(torch.nn.functional.softplus(-margin)*pw).sum()+policy['gpu']['correction_MSE_penalty']*correction[ix].square().mean()
                loss.backward();opt.step()
            for actual,expected in [(net[0].weight,'hidden_weight'),(net[0].bias,'hidden_bias'),(net[2].weight,'output_weight'),(net[2].bias,'output_bias')]:
                np.testing.assert_allclose(actual.detach().cpu().numpy(),model[expected],rtol=0,atol=1e-6)
            # Independent float64 NumPy implementation of exported GPU network.
            h=z@model['hidden_weight'].T+model['hidden_bias']
            h=.5*h*(1+np.tanh(np.sqrt(2/np.pi)*(h+.044715*h**3)))
            correction=.25*np.tanh((h@model['output_weight'].T+model['output_bias']).ravel())
            predictions[held]=anchor[held]+correction[held]
            assert np.abs(correction).max()<=.25
            assert result['folds'][fold]['epochs']==200 and result['folds'][fold]['GPU_peak_memory_allocated']>0
            assert result['folds'][fold]['model_sha256']==sha(out/f'gpu-fold{fold}.npz')
        scores=pd.read_parquet(out/'gpu-oof-scores.parquet');np.testing.assert_allclose(scores.oof_score,predictions,rtol=0,atol=1e-6)
        lists={mid:g.sort_values('parent_rank').smiles.tolist() for mid,g in rows.groupby('molecule_id',sort=False)}
        for (mid,_),g in rows.groupby(['molecule_id','formula'],sort=False):
            if len(g)>1 and g.available.all():
                order=sorted(g.index,key=lambda i:(-predictions[i],int(rows.parent_rank.iloc[i])))
                for rank,text in zip(g.parent_rank,rows.smiles.iloc[order]):lists[mid][int(rank)-1]=text
        actual=pd.read_csv(out/'gpu-oof.csv',dtype=str);assert actual.molecule_id.tolist()==q.molecule_id.tolist()
        assert actual.smiles.tolist()==[';'.join(lists[mid]) for mid in q.molecule_id]
        stored=np.load(out/'gpu-oof-ranks.npz');ranks={n:[] for n in ['P01','P16','physical_parent','cpu']}
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
        intersections={}
        for (cohort,source,mode),g in q.groupby(['cohort','source','mode'],sort=False):
            p=stored['P01'][g.index];c=stored['cpu'][g.index]
            d=sum((Fraction(1,int(v)) if v else Fraction(0))-(Fraction(1,int(u)) if u else Fraction(0)) for u,v in zip(p,c))/len(g)
            intersections['/'.join([cohort,source,mode])]=dict(n=len(g),delta=float(d),nonnegative=d>=0)
        p=stored['P01'];c=stored['cpu'];delta=np.where(c>0,1/np.maximum(c,1),0)-np.where(p>0,1/np.maximum(p,1),0)
        group_delta=q.assign(delta=delta).groupby('chemistry_group').agg(total=('delta','sum'),n=('delta','size'))
        rng=np.random.default_rng(20261166);indices=rng.integers(0,len(group_delta),size=(10000,len(group_delta)))
        boot=group_delta.total.to_numpy()[indices].sum(1)/group_delta.n.to_numpy()[indices].sum(1)
        write(target,dict(completed_utc=stamp(),source_sha256=sha(__file__),result_sha256=sha(out/'gpu-oof-results.json'),queries=869,acquisitions=3452,
            parent_evidence_independently_reconstructed=True,old87_features_and_all_original_inputs_unchanged=True,
            source_mode_intersections=intersections,improved_queries=int((delta>0).sum()),harmed_queries=int((delta<0).sum()),
            chemistry_group_bootstrap_95_interval=np.quantile(boot,[.025,.975]).tolist(),bootstrap_limitation='Adaptive USED OOF, not a fresh prospective uncertainty guarantee',
            query_chemistry_edges_never_cross_folds=True,train_only_scaling_exact=True,convex_training_objective_gradient_maxima=gradients,
            actual_saved_OOF_scores_and_lists_exact=True,all_original_source_mode_fraction_points_exact=True,absent_truths_zero=True,
            GPU_gate_passed=result['GPU_development_gate_passed'],all5_GPU_training_replays_exact=True,exported_NumPy_inference_scores_match=True,C13_read=False,scoring_entries=0,prospective_qualification=False))
        print(tag,'INDEPENDENT GPU OOF AUDIT PASS; gate',result['GPU_development_gate_passed'],'gradientmax',max(gradients),flush=True)


if __name__=='__main__':main()
