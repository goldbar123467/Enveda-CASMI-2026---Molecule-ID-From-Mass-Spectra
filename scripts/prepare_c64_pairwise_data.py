"""Prepare all869USED cases for chemistry-group OOF direct isomer ranking."""
import os
for n in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(n,'2')
import torch
import hashlib,json,types
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem,DataStructs,RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from c45_complete_parent import ROOT,read,write,sha,stamp,module
from legacy_formate_compat import patch_chem_source
OUT=ROOT/'outputs/local/c64-pairwise-oof-20261112'
POLICY=ROOT/'configs/c64-physics-pairwise-oof-20261112.json'
PANELS=[
    ('C39',352,'outputs/local/c45-complete-parent-validation-20261112','outputs/local/c39-source-supported-validation-20261111/labels.csv','outputs/forks/c41r3-full-masked-validation-v1','outputs/local/c56-bounded-semantic-heads-20261112/C39-predictions','C39-parent','C39-masked-reference'),
    ('C51',35,'outputs/local/c51-complete-auxiliary-20261112','outputs/local/c48-auxiliary-holdout-20261112/labels.csv','outputs/forks/c50-full-masked-validation-v1','outputs/local/c56-bounded-semantic-heads-20261112/C51-predictions','auxiliary-parent','auxiliary-masked-reference'),
    ('C53',482,'outputs/local/c58-complete-target-20261112','outputs/local/c53-target-holdout-20261112/labels.csv','outputs/forks/c57-full-masked-validation-v1','outputs/local/c58-complete-target-20261112/head-predictions','target-parent','target-masked-reference')]


def main():
    policy=read(POLICY);OUT.mkdir(exist_ok=True);assert not (OUT/'data-freeze.json').exists()
    assert not read(ROOT/'outputs/local/c58-complete-target-20261112/source-results.json')['passed_pair']
    assert not (ROOT/'outputs/local/c13-confirmation-20261110/evaluation-consumption.json').exists()
    RDLogger.DisableLog('rdApp.*');torch.set_num_threads(2)
    legacy=ROOT/'outputs/local/c37-selection-trace-20261111/pinned-public-source/chem.py';chem=types.ModuleType('c64_mass')
    exec(compile(patch_chem_source(legacy.read_text(encoding='utf8')),str(legacy),'exec'),chem.__dict__)
    identity_fn=module('c64_official_identity','c09e_features.py').structure
    identities={};query_records=[];candidate_records=[];inputs=[POLICY,Path(__file__),legacy,ROOT/'scripts/legacy_formate_compat.py'];physical_changes=[]
    aliases={'[M+H-H2O]+':'[M-H2O+H]+','[M+H-2H2O]+':'[M-2H2O+H]+'}
    for cohort,count,base_path,label_path,core_path,cache_path,parent_name,ref_name in PANELS:
        base,core,cache=ROOT/base_path,ROOT/core_path,ROOT/cache_path
        original=pd.read_csv(base/'P01.csv',dtype=str,keep_default_na=False);p16=pd.read_csv(base/'P16.csv',dtype=str,keep_default_na=False)
        assert original.molecule_id.tolist()==p16.molecule_id.tolist() and len(original)==count
        labels=pd.read_csv(ROOT/label_path,dtype=str).set_index('molecule_id').loc[original.molecule_id]
        q=pd.read_parquet(base/'queries.parquet');groups=dict(tuple(q.groupby('molecule_id',sort=False)))
        cached=pd.read_parquet(cache/'candidate-identities.parquet')
        for row in cached.itertuples(index=False):identities[row.smiles]=(row.key,row.formula)
        def identity(text):
            if text not in identities:
                key,_,formula,*_=identity_fn(text);identities[text]=(key,formula)
            return identities[text]
        raw=pd.read_parquet(core/'c41_neural_p01.parquet');public=pd.read_parquet(core/'neural_public.parquet');ours=pd.read_parquet(core/'neural_ours.parquet')
        assert not raw.duplicated(['molecule_id','metric_key']).any()
        raw=raw.set_index(['molecule_id','metric_key']);public=public.set_index(['molecule_id','metric_key']);ours=ours.set_index(['molecule_id','metric_key'])
        matches=pd.read_parquet(base/parent_name/'c09e_matches.parquet');references=pd.read_parquet(base/ref_name/'ext_meta.parquet')
        for i,row in enumerate(original.itertuples(index=False)):
            mid=row.molecule_id;g=groups[mid];a=row.smiles;b=p16.smiles.iloc[i];chosen=b
            if a!=b:
                hit=matches[matches.molecule_id.eq(mid)&matches.view.eq('all_reference')].iloc[0];reference=references.iloc[int(hit.reference_row)]
                query_mass=chem.neutral_mass(g.precursor_mz.to_numpy(),g.adduct.to_numpy());query_mass=query_mass[np.isfinite(query_mass)]
                adduct=aliases.get(reference.adduct,reference.adduct)
                ref_mass=float(chem.neutral_mass(np.array([reference.precursor_mz]),np.array([adduct]))[0])
                nm=float(np.median(query_mass)) if len(query_mass) else float('nan')
                consistent=bool(np.isfinite(nm) and np.isfinite(ref_mass) and abs(nm-ref_mass)<=max(.002,nm*10e-6))
                if not consistent:chosen=a
                physical_changes.append(dict(cohort=cohort,molecule_id=mid,adduct_mass_consistent=consistent,reference_adduct=reference.adduct,
                    reference_neutral_mass=ref_mass,query_neutral_mass=nm,reverted_to_P01=not consistent))
            mode='mixed' if g.ionization_mode.nunique()>1 else g.ionization_mode.iloc[0]
            qr=dict(molecule_id=mid,cohort=cohort,source=labels.source.iloc[i],mode=mode,key=labels.key.iloc[i],smiles=labels.smiles.iloc[i],formula=labels.formula.iloc[i],
                P01=a,P16=b,physical_parent=chosen,acquisitions=len(g),positive_fraction=float(g.ionization_mode.eq('positive').mean()),
                H_fraction=float(g.adduct.eq('[M+H]+').mean()),precursor=float(g.precursor_mz.median())/1000)
            query_records.append(qr)
            for rank,text in enumerate(chosen.split(';'),1):
                key,formula=identity(text);lookup=(mid,key);available=lookup in raw.index
                record=dict(molecule_id=mid,key=key,smiles=text,formula=formula,parent_rank=rank,available=available,target=int(key==labels.key.iloc[i]))
                for field in ['library_max','proxy_gl','ice','gl','main_rank','prefusion_rank','engine2_rank','pubchem_rank','neural_cos','neural_fp','neural_score','neural_std','n_spectra','n_adducts']:
                    record[field]=float(raw.loc[lookup,field]) if available else float('nan')
                for name,frame in [('public',public),('ours',ours)]:
                    for field in ['neural_cos','neural_fp','neural_score','neural_std','neural_ok']:
                        record[name+'_'+field]=float(frame.loc[lookup,field]) if lookup in frame.index else float('nan')
                candidate_records.append(record)
        inputs += [base/n for n in ['P01.csv','P16.csv','queries.parquet']]+[ROOT/label_path,cache/'candidate-identities.parquet']
        inputs += [core/n for n in ['c41_neural_p01.parquet','neural_public.parquet','neural_ours.parquet']]+[base/parent_name/'c09e_matches.parquet',base/ref_name/'ext_meta.parquet']
        print('C64 ASSEMBLED',cohort,count,flush=True)
    queries=pd.DataFrame(query_records);candidates=pd.DataFrame(candidate_records)
    assert len(queries)==queries.molecule_id.nunique()==869 and queries.acquisitions.sum()==3452
    # Query chemistry components, not random spectrum/candidate folds.
    mol=[Chem.MolFromSmiles(s) for s in queries.smiles];assert all(m is not None for m in mol)
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048);fp=[gen.GetFingerprint(m) for m in mol]
    scaffold=[MurckoScaffold.MurckoScaffoldSmiles(mol=m,includeChirality=False) for m in mol];roots=list(range(len(queries)))
    def find(i):
        if roots[i]!=i:roots[i]=find(roots[i])
        return roots[i]
    for i in range(len(queries)):
        sim=DataStructs.BulkTanimotoSimilarity(fp[i],fp[:i])
        for j in range(i):
            if queries.key.iloc[i]==queries.key.iloc[j] or queries.formula.iloc[i]==queries.formula.iloc[j] or bool(scaffold[i] and scaffold[i]==scaffold[j]) or sim[j]>=.7:roots[find(i)]=find(j)
    queries['chemistry_group']=[find(i) for i in range(len(queries))]
    totals=[0]*5;assignment={};components=queries.groupby('chemistry_group',sort=False)
    order=sorted(components.groups,key=lambda k:hashlib.sha256((str(policy['seed'])+'|'+min(queries.key.iloc[components.groups[k]])).encode()).hexdigest())
    for group in order:
        fold=min(range(5),key=lambda k:(totals[k],k));assignment[group]=fold;totals[fold]+=len(components.groups[group])
    queries['fold']=queries.chemistry_group.map(assignment);assert queries.groupby('chemistry_group').fold.nunique().eq(1).all()
    qr=queries.set_index('molecule_id');feature_names=['parent_log_rank'];columns=[-np.log(candidates.parent_rank.to_numpy(float))]
    raw_names=[n for n in candidates.columns if n not in ['molecule_id','key','smiles','formula','parent_rank','available','target']]
    for name in raw_names:
        values=candidates[name].to_numpy(float);finite=np.isfinite(values);safe=np.where(finite,values,0.)
        columns.extend([safe,finite.astype(float)]);feature_names.extend([name,name+'_finite'])
        tmp=pd.Series(safe).groupby(candidates.molecule_id).transform(lambda x:(x-x.mean())/max(float(x.std(ddof=0)),1e-6)).to_numpy()
        columns.append(np.clip(tmp,-10,10));feature_names.append(name+'_query_z')
        if name in ['ice','gl','neural_score','public_neural_score','ours_neural_score']:
            for meta in ['positive_fraction','H_fraction']:
                columns.append(np.clip(tmp,-10,10)*candidates.molecule_id.map(qr[meta]).to_numpy());feature_names.append(name+'_z_x_'+meta)
    for name in ['acquisitions','positive_fraction','H_fraction','precursor']:
        columns.append(candidates.molecule_id.map(qr[name]).to_numpy(float));feature_names.append('query_'+name)
    x=np.stack(columns,axis=1).astype(np.float32);assert np.isfinite(x).all()
    candidates['query_index']=candidates.molecule_id.map({mid:i for i,mid in enumerate(queries.molecule_id)});candidates['fold']=candidates.molecule_id.map(qr.fold)
    assert not candidates.duplicated(['molecule_id','key']).any()
    np.save(OUT/'features.npy',x);queries.to_parquet(OUT/'queries.parquet',index=False);candidates.to_parquet(OUT/'candidates.parquet',index=False)
    write(OUT/'feature-names.json',feature_names);write(OUT/'physical-parent-changes.json',physical_changes)
    report=dict(prepared_utc=stamp(),source_sha256=sha(__file__),policy_sha256=sha(POLICY),queries=869,acquisitions=3452,candidates=len(candidates),features=x.shape[1],
        chemistry_groups=queries.chemistry_group.nunique(),fold_queries=totals,all_original_queries_retained=True,missing_truths_will_score_zero=True,
        C53_now_USED=True,C13_read=False,no_C56_fitted_projection_reused=True,model_fitted=False,scoring_entries=0,
        files_sha256={str(p.relative_to(ROOT)):sha(p) for p in inputs+[OUT/n for n in ['features.npy','queries.parquet','candidates.parquet','feature-names.json','physical-parent-changes.json']]})
    write(OUT/'data-freeze.json',report);print('C64 ALL869 OOF DATA READY',json.dumps({k:v for k,v in report.items() if k!='files_sha256'}),flush=True)


if __name__=='__main__':main()
