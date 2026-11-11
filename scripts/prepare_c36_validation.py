"""Freeze CPU-only serving features and chemistry groups from consumed panels."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2'); os.environ.setdefault('MKL_NUM_THREADS','2')
import functools,hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
from rdkit import Chem,DataStructs,RDLogger,rdBase
from rdkit.Chem import rdFingerprintGenerator,rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from validation_repair import folds_for,relative

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/local/c36-validation-repair-20261111'
CONFIG=ROOT/'configs/c36-cpu-validation-20261111.json'
sys.path.insert(0,str(ROOT/'outputs/local/kit'))
import features


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,value):
    Path(p).write_text(json.dumps(value,indent=2),encoding='utf-8')


def verify():
    contract=json.loads((OUT/'input-contract.json').read_text())
    assert sha(CONFIG)==contract['config_sha256']
    for name,pin in contract['source_sha256'].items():assert sha(ROOT/'scripts'/name)==pin,name
    for section in ['inputs','routes']:
        for name,r in contract[section].items():assert sha(ROOT/r['path'])==r['sha256'],name
    frozen=json.loads((OUT/'feature-freeze.json').read_text())
    assert frozen['input_contract_sha256']==sha(OUT/'input-contract.json')
    for name,pin in frozen['files_sha256'].items():assert sha(OUT/name)==pin,name
    return contract,frozen


def load_blocks():
    verify();schema=json.loads((OUT/'feature-schema.json').read_text());names=schema['feature_names']
    evidence=pd.read_parquet(OUT/'features.parquet');metadata=pd.read_csv(OUT/'queries.csv',dtype={'group':str})
    groups={mid:g.sort_values('baseline_rank',kind='stable') for mid,g in evidence.groupby('molecule_id',sort=False)}
    blocks=[]
    for row in metadata.itertuples(index=False):
        g=groups[row.molecule_id]
        blocks.append(dict(mid=row.molecule_id,cohort=row.cohort,mode=row.mode,group=row.group,truth=row.truth,
           raw_alias=bool(row.raw_alias_flag),outer_fold=int(row.outer_fold),library=float(row.library),
           keys=g.metric_key.tolist(),formulas=g.formula.tolist(),smiles=g.smiles.tolist(),x=g[names].to_numpy(float)))
    return blocks,names


def main():
    pa.set_cpu_count(2);RDLogger.DisableLog('rdApp.*')
    assert rdBase.rdkitVersion=='2026.03.3'
    assert (ROOT/'.local/compute-paused.json').is_file()
    assert not (OUT/'feature-freeze.json').exists() and not (OUT/'input-contract.json').exists()
    cfg=json.loads(CONFIG.read_text())
    paths={
      'parent':'outputs/local/c21-negative-tail-20261110/parent.csv',
      'queries':'outputs/local/c21-negative-tail-20261110/queries.parquet',
      'positive_features':'outputs/forks/c10-casmi2022-validation-v1/c10_neural_p01.parquet',
      'negative_features':'outputs/forks/c20-negative-development-v2/c20_neural_p01.parquet',
      'primary_index':'outputs/local/c21-negative-tail-20261110/fingerprint-smiles.parquet',
      'deep_index':'outputs/local/c12-deep-candidates-20261110/all/fingerprint-smiles.parquet',
      'deep_pool':'outputs/local/c12-deep-candidates-20261110/all/pool.parquet',
      'positive_labels':'outputs/local/c10-casmi2022-20261110/labels.csv',
      'negative_labels':'outputs/local/c20-negative-development-20261110/labels.csv',
      'P14':'outputs/local/c21-negative-tail-20261110/P14.csv',
      'P15':'outputs/local/c21-negative-tail-20261110/P15.csv',
      'P18':'outputs/local/two-past-trials-20261110/C22-peak-dose_allmode.csv',
      'P19':'outputs/local/two-past-trials-20261110/C18-peak-drop_allmode.csv',
      'stress_parent':'outputs/forks/c09-clean-validation-v2/c09_clean_p01.csv',
      'stress_features':'outputs/forks/c09-clean-validation-v2/c09_clean_neural_p01.parquet',
      'stress_queries':'outputs/forks/c09-clean-validation-v2/c09_clean_queries.parquet',
      'stress_labels':'outputs/publish/c09-validation-20261109/labels.csv',
      'stress_P14':'outputs/forks/c09-clean-validation-v2/c09_clean_p14.csv',
      'stress_aliases':'outputs/forks/c09-spectrum-audit-v3/c09_spectrum_matches.csv',
      'prior_primary_consumption':'outputs/local/c21-negative-tail-20261110/development.json',
      'prior_positive_consumption':'outputs/local/c12-deep-candidates-20261110/confirmation.json',
      'prior_stress_consumption':'outputs/local/c09-followup-evaluation-20261109/confirmation.json'}
    routes={}
    for label,folder in [('positive','outputs/forks/c10-candidate-pools-v1'),('negative','outputs/forks/c20-negative-development-v2')]:
        for name in ['eng_lists.json','pc_lists.json','ice_work/ice_in.json','ice_work/ice_out.json','gl_work/gl_in.json','gl_work/gl_out.json']:
            routes[label+':'+name]=folder+'/'+name
    routes['positive:n1_work/n1_in.json']='outputs/forks/c10-candidate-pools-v1/n1_work/n1_in.json'
    routes['negative:recovered_engine']='outputs/forks/c20-negative-development-v2/c20_engine2_recovered_lists.json'
    sources=['prepare_c36_validation.py','audit_c36_candidate_losses.py','evaluate_c36_isomers.py','validation_repair.py']
    contract=dict(prepared_utc=datetime.now(timezone.utc).isoformat(),config_sha256=sha(CONFIG),
      source_sha256={n:sha(ROOT/'scripts'/n) for n in sources},feature_identity_source_sha256=sha(ROOT/'outputs/local/kit/features.py'),
      inputs={name:dict(path=p,sha256=sha(ROOT/p)) for name,p in paths.items()},
      routes={name:dict(path=p,sha256=sha(ROOT/p)) for name,p in routes.items()},
      fresh_reserved_labels_read=False,gpu_work=False,cloud_runs=False,submissions=False,
      selection='Freeze prior to all new variant metrics; no query filtering based on control or candidate outcomes')
    write(OUT/'input-contract.json',contract)
    assert json.loads((ROOT/paths['prior_primary_consumption']).read_text())['queries']==240
    assert json.loads((ROOT/paths['prior_positive_consumption']).read_text())['queries']==120
    # The stress80 result is historical/consumed; never open C13 labels here.
    assert (ROOT/paths['prior_stress_consumption']).is_file()
    parent=pd.read_csv(ROOT/paths['parent'],dtype=str,keep_default_na=False)
    stress=pd.read_csv(ROOT/paths['stress_parent'],dtype=str,keep_default_na=False)
    raw=pd.concat([pd.read_parquet(ROOT/paths[n]) for n in ['positive_features','negative_features','stress_features']],ignore_index=True)
    assert not raw.duplicated(['molecule_id','metric_key']).any()
    known={}
    for name in ['primary_index','deep_index']:
        f=pd.read_parquet(ROOT/paths[name]);known.update(dict(zip(f.smiles,f.metric_key)))
    known.update(dict(zip(raw.smiles,raw.metric_key)))
    formula_cache={}
    @functools.lru_cache(maxsize=40000)
    def canonical(s):
        if s in known:
            # Cached keys were already independently checked with this exact
            # RDKit/identity source. Formula/mass/scaffold need no tautomer sweep.
            mol=Chem.MolFromSmiles(s);assert mol is not None
            Chem.RemoveStereochemistry(mol)
            value=(known[s],Chem.MolToSmiles(mol),rdMolDescriptors.CalcMolFormula(mol),rdMolDescriptors.CalcExactMolWt(mol))
        else:value=features.structure(s)
        known[s]=value[0];formula_cache[s]=value[2]
        return value
    query=pd.concat([pd.read_parquet(ROOT/paths['queries']),pd.read_parquet(ROOT/paths['stress_queries'])],ignore_index=True)
    mode=query.drop_duplicates('molecule_id').set_index('molecule_id').ionization_mode.astype(str).to_dict()
    lookup=raw.set_index(['molecule_id','metric_key'])
    feature_names=['inverse_baseline','library_inverse_baseline']
    feature_names += ['inverse_'+n for n in ['main','prefusion','engine2','pubchem']]
    channels=['neural_score','neural_cos','neural_fp','ice','gl','proxy_gl']
    for c in channels:feature_names += [c+'_present',c+'_z',c+'_inverse_rank']
    feature_names += ['ice_neural_agreement','gl_neural_agreement']
    records=[];meta=[];missing=0
    for cohort,frame in [('primary',parent),('source_stress',stress)]:
        for number,row in enumerate(frame.itertuples(index=False)):
            smis=row.smiles.split(';');keys=[];forms=[];series=[]
            for s in smis:
                value=canonical(s);k=value[0];keys.append(k);forms.append(value[2])
                assert (row.molecule_id,k) in lookup.index,'Parent candidate missing cached serving features'
                series.append(lookup.loc[(row.molecule_id,k)])
            assert len(keys)==len(set(keys)) and 1<=len(keys)<=25
            g=pd.DataFrame(series).reset_index(drop=True);lib=float(g.library_max.iloc[0])
            assert np.isfinite(lib) and g.library_max.nunique()==1
            inv=1/np.arange(1,len(smis)+1,dtype=float);columns=[inv,lib*inv]
            for c in ['main','prefusion','engine2','pubchem']:
                v=pd.to_numeric(g[c+'_rank'],errors='coerce').to_numpy(float)
                z=np.zeros(len(v));valid=np.isfinite(v)&(v>0);z[valid]=1/v[valid];columns.append(z)
            rel={}
            for c in channels:
                rel[c]=relative(pd.to_numeric(g[c],errors='coerce').to_numpy(float));columns.extend(rel[c])
            columns += [rel['ice'][1]*rel['neural_score'][1],rel['gl'][1]*rel['neural_score'][1]]
            matrix=np.stack(columns,axis=1);assert np.isfinite(matrix).all() and matrix.shape==(len(keys),len(feature_names))
            for j,(s,k,f) in enumerate(zip(smis,keys,forms)):
                record=dict(cohort=cohort,molecule_id=row.molecule_id,smiles=s,metric_key=k,formula=f,baseline_rank=j+1,library_max=lib)
                record.update({name:float(matrix[j,q]) for q,name in enumerate(feature_names)});records.append(record)
            meta.append(dict(cohort=cohort,molecule_id=row.molecule_id,mode=mode[row.molecule_id],library=lib))
            if (number+1)%40==0:print('C36 CACHED FEATURES',cohort,number+1,flush=True)
    evidence=pd.DataFrame(records);evidence.to_parquet(OUT/'features.parquet',index=False)
    write(OUT/'feature-schema.json',dict(feature_names=feature_names,physical_features=[n for n in feature_names if n.startswith(('ice_','gl_','proxy_gl_'))],
          inputs=contract['inputs'],label_free_feature_preparation=True,formula_from_candidate_only=True))
    # Features and candidate membership are sealed before any targets are joined.
    write(OUT/'label-free-feature-freeze.json',dict(features_sha256=sha(OUT/'features.parquet'),schema_sha256=sha(OUT/'feature-schema.json'),quality_metrics_computed=False))
    labels=pd.concat([pd.read_csv(ROOT/paths[n],dtype=str) for n in ['positive_labels','negative_labels','stress_labels']],ignore_index=True)
    assert len(labels)==320 and labels.key.nunique()==320 and labels.molecule_id.is_unique
    labels=labels.set_index('molecule_id')
    metadata=pd.DataFrame(meta);assert len(metadata)==320 and metadata.molecule_id.is_unique
    assert set(metadata.molecule_id)==set(labels.index)
    metadata['truth']=metadata.molecule_id.map(labels.key)
    metadata['truth_smiles']=metadata.molecule_id.map(labels.smiles)
    mols=[];fingerprints=[];scaffolds=[];formulas=[];masses=[];neutral_masses=[];peak_signatures=[]
    generator=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
    for r in metadata.itertuples(index=False):
        value=canonical(r.truth_smiles);assert value[0]==r.truth
        mol=Chem.MolFromSmiles(value[1]);mols.append(mol);fingerprints.append(generator.GetFingerprint(mol))
        scaffolds.append(MurckoScaffold.MurckoScaffoldSmiles(mol=mol,includeChirality=False));formulas.append(value[2]);masses.append(value[3])
        neutral_masses.append(float(query[query.molecule_id.eq(r.molecule_id)].nm.median()))
        signatures=set()
        for q in query[query.molecule_id.eq(r.molecule_id)].itertuples(index=False):
            mz=np.asarray(q.ms2_mzs,float);it=np.asarray(q.ms2_normalized_intensities,float)
            take=np.isfinite(mz)&np.isfinite(it)&(mz>0)&(it>0);mz,it=mz[take],it[take]
            order=np.argsort(mz,kind='stable');payload=np.stack([np.round(mz[order],3),np.round(it[order]/max(float(it.max()),1e-8),4)],1)
            signatures.add(hashlib.sha256(payload.tobytes()).hexdigest())
        peak_signatures.append(signatures)
    links=list(range(320))
    def find(i):
        while links[i]!=i:links[i]=links[links[i]];i=links[i]
        return i
    def union(i,j):links[find(i)]=find(j)
    edge_counts=dict(scaffold=0,formula=0,tanimoto=0,acquisition=0)
    for i in range(320):
        similarities=DataStructs.BulkTanimotoSimilarity(fingerprints[i],fingerprints[:i])
        for j,sim in enumerate(similarities):
            flags=dict(scaffold=bool(scaffolds[i]) and scaffolds[i]==scaffolds[j],formula=formulas[i]==formulas[j],
                       tanimoto=sim>=.70,acquisition=bool(peak_signatures[i]&peak_signatures[j]))
            if any(flags.values()):
                union(i,j)
                for name,value in flags.items():edge_counts[name]+=int(value)
    metadata['group']=[str(find(i)) for i in range(320)]
    metadata['truth_formula']=formulas;metadata['truth_mass']=masses;metadata['query_neutral_mass']=neutral_masses
    aliases=pd.read_csv(ROOT/paths['stress_aliases'],dtype=str)
    alias_ids=set(aliases.molecule_id.astype(str));assert len(alias_ids)==20
    metadata['raw_alias_flag']=metadata.molecule_id.isin(alias_ids)
    primary_meta=metadata[metadata.cohort.eq('primary')].reset_index(drop=True)
    blocks=[dict(group=r.group,mode=r.mode) for r in primary_meta.itertuples(index=False)]
    outer=folds_for(blocks,list(range(240)),cfg['outer_folds'],cfg['split_salt'])
    metadata['outer_fold']=-1
    for i,(position,_) in enumerate(metadata[metadata.cohort.eq('primary')].iterrows()):metadata.loc[position,'outer_fold']=outer[i]
    metadata.to_csv(OUT/'queries.csv',index=False)
    pd.DataFrame(dict(smiles=list(known),metric_key=list(known.values()))).to_parquet(OUT/'identity-cache.parquet',index=False)
    report=dict(completed_utc=datetime.now(timezone.utc).isoformat(),input_contract_sha256=sha(OUT/'input-contract.json'),
      files_sha256={n:sha(OUT/n) for n in ['features.parquet','feature-schema.json','label-free-feature-freeze.json','queries.csv','identity-cache.parquet']},
      primary_queries=240,stress_queries=80,primary_groups=int(primary_meta.group.nunique()),
      largest_primary_group=int(primary_meta.group.value_counts().max()),outer_fold_sizes={str(i):list(outer.values()).count(i) for i in range(5)},
      grouping_edges=edge_counts,stress_raw_alias_flagged=20,stress_clean=60,
      unknown_foundation_model_pretraining_exposure=True,fresh_reserved_labels_read=False,gpu_work=False)
    write(OUT/'feature-freeze.json',report);print('DONE C36 LABEL-FREE FEATURE AND GROUP FREEZE',json.dumps(report),flush=True)


if __name__=='__main__':main()
