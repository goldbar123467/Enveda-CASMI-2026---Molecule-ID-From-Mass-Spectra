"""Outcome-blind deployment of the C66/C67 direct isomer rankers.

Inputs are complete parent predictions, original acquisitions and frozen public
core evidence. Query truth, source labels and development partitions are never
accepted. Unknown candidates keep their complete same-formula parent group.
"""
import numpy as np
import pandas as pd

RAW_FIELDS=['library_max','proxy_gl','ice','gl','main_rank','prefusion_rank',
    'engine2_rank','pubchem_rank','neural_cos','neural_fp','neural_score',
    'neural_std','n_spectra','n_adducts']
NEURAL_FIELDS=['neural_cos','neural_fp','neural_score','neural_std','neural_ok']
META=['acquisitions','positive_fraction','H_fraction','precursor']


def physical_parent(p01,p16,acquisitions,matches,references,neutral_mass):
    """Keep an external promotion only with compatible inferred neutral mass."""
    assert p01.molecule_id.tolist()==p16.molecule_id.tolist()
    result=p16.copy();events=[]
    groups=dict(tuple(acquisitions.groupby('molecule_id',sort=False)))
    aliases={'[M+H-H2O]+':'[M-H2O+H]+','[M+H-2H2O]+':'[M-2H2O+H]+'}
    for i,row in p01.iterrows():
        if row.smiles==p16.smiles.iloc[i]:continue
        g=groups[row.molecule_id]
        hit=matches[matches.molecule_id.eq(row.molecule_id)&matches.view.eq('all_reference')].iloc[0]
        ref=references.iloc[int(hit.reference_row)]
        qm=neutral_mass(g.precursor_mz.to_numpy(),g.adduct.to_numpy());qm=qm[np.isfinite(qm)]
        rm=float(neutral_mass(np.array([ref.precursor_mz]),np.array([aliases.get(ref.adduct,ref.adduct)]))[0])
        nm=float(np.median(qm)) if len(qm) else float('nan')
        consistent=bool(np.isfinite(nm) and np.isfinite(rm) and abs(nm-rm)<=max(.002,nm*10e-6))
        if not consistent:result.at[i,'smiles']=row.smiles
        events.append(dict(molecule_id=row.molecule_id,mass_consistent=consistent,query_mass=nm,reference_mass=rm))
    return result,events


def assemble(parent,acquisitions,raw,public,ours,pop,identity):
    """Return identifier-free101features and a separate candidate row index."""
    assert not raw.duplicated(['molecule_id','metric_key']).any()
    raw=raw.copy();raw['nr']=raw.groupby('molecule_id',sort=False).neural_score.rank(method='first',ascending=False)
    raw=raw.set_index(['molecule_id','metric_key'])
    public=public.set_index(['molecule_id','metric_key']);ours=ours.set_index(['molecule_id','metric_key'])
    for frame in [public,ours]:assert frame.index.is_unique
    groups=dict(tuple(acquisitions.groupby('molecule_id',sort=False)));records=[];meta=[]
    for item in parent.itertuples(index=False):
        q=groups[item.molecule_id]
        meta.append(dict(molecule_id=item.molecule_id,acquisitions=len(q),positive_fraction=float(q.ionization_mode.eq('positive').mean()),
            H_fraction=float(q.adduct.eq('[M+H]+').mean()),precursor=float(q.precursor_mz.median())/1000))
        for rank,text in enumerate(item.smiles.split(';'),1):
            key,_,formula,*_=identity(text);ix=(item.molecule_id,key);available=ix in raw.index
            row=dict(molecule_id=item.molecule_id,key=key,smiles=text,formula=formula,parent_rank=rank,available=available)
            for field in RAW_FIELDS:row[field]=float(raw.loc[ix,field]) if available else float('nan')
            for tag,frame in [('public',public),('ours',ours)]:
                for field in NEURAL_FIELDS:row[tag+'_'+field]=float(frame.loc[ix,field]) if ix in frame.index else float('nan')
            row['br']=float(raw.loc[ix,'baseline_rank']) if available else 0.
            row['nr']=float(raw.loc[ix,'nr']) if available else 0.
            records.append(row)
    rows=pd.DataFrame(records);qr=pd.DataFrame(meta).set_index('molecule_id')
    names=['parent_log_rank'];columns=[-np.log(rows.parent_rank.to_numpy(float))]
    raw_names=RAW_FIELDS+[tag+'_'+field for tag in ['public','ours'] for field in NEURAL_FIELDS]
    for name in raw_names:
        value=rows[name].to_numpy(float);finite=np.isfinite(value);safe=np.where(finite,value,0.)
        z=pd.Series(safe).groupby(rows.molecule_id).transform(lambda a:(a-a.mean())/max(float(a.std(ddof=0)),1e-6)).to_numpy()
        z=np.clip(z,-10,10);columns.extend([safe,finite.astype(float),z]);names.extend([name,name+'_finite',name+'_query_z'])
        if name in ['ice','gl','neural_score','public_neural_score','ours_neural_score']:
            for field in ['positive_fraction','H_fraction']:
                columns.append(z*rows.molecule_id.map(qr[field]).to_numpy());names.append(name+'_z_x_'+field)
    for field in META:columns.append(rows.molecule_id.map(qr[field]).to_numpy(float));names.append('query_'+field)
    pop=pop.drop_duplicates('inchikey14',keep='last').set_index('inchikey14')
    popularity=rows.key.map(pop['pop']).fillna(0).to_numpy(float)
    natural=rows.key.map(pop[['np_coconut','np_lotus','src_natural_product']].any(axis=1).astype(int)).fillna(0).to_numpy(float)
    br=rows.br.to_numpy(float);nr=rows.nr.to_numpy(float)
    added=dict(public_baseline_rank=br,parent_neural_rank=nr,public_rank_RRF=1/(3+br),parent_neural_RRF=.75/(3+nr),
        documentation_prior=popularity/20,natural_product_prior=natural,parent_tail_score=1/(3+br)+.75/(3+nr)+.005*popularity+.02*natural)
    for name,value in added.items():
        z=pd.Series(value).groupby(rows.molecule_id).transform(lambda a:(a-a.mean())/max(float(a.std(ddof=0)),1e-6)).to_numpy()
        columns.extend([value,np.clip(z,-10,10)]);names.extend([name,name+'_query_z'])
    x=np.stack(columns,axis=1).astype(np.float32);assert x.shape[1]==101 and np.isfinite(x).all()
    return rows,x,names


def predict(rows,x,model,gpu=False):
    z=np.clip((x.astype(np.float64)-model['mean'])/model['std'],-10,10)
    score=-np.log(rows.parent_rank.to_numpy(float))+z@model['weights']
    if gpu:
        h=z@model['hidden_weight'].T+model['hidden_bias']
        h=.5*h*(1+np.tanh(np.sqrt(2/np.pi)*(h+.044715*h**3)))
        score+=.25*np.tanh((h@model['output_weight'].T+model['output_bias']).ravel())
    assert np.isfinite(score).all()
    lists={mid:g.sort_values('parent_rank').smiles.tolist() for mid,g in rows.groupby('molecule_id',sort=False)}
    for (mid,_),g in rows.groupby(['molecule_id','formula'],sort=False):
        if len(g)<=1 or not g.available.all():continue
        order=sorted(g.index,key=lambda i:(-score[i],int(rows.parent_rank.iloc[i])))
        for rank,text in zip(g.parent_rank,rows.smiles.iloc[order]):lists[mid][int(rank)-1]=text
    return lists,score
