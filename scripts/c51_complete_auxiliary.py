"""Label-outcome-blind reference masking and exact frozen P16 replay.

Only the mask phase reads auxiliary structures for the predeclared chemistry holdout.
No function matches a prediction to query truth or computes a quality metric.
"""
import os
for _key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(_key,'2')
import argparse,copy,hashlib,importlib.util,json,subprocess,sys
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import pyarrow as pa
from rdkit import Chem,DataStructs,RDLogger,rdBase
from rdkit.Chem.Scaffolds import MurckoScaffold

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/local/c51-complete-auxiliary-20261112'
CODE=ROOT/'outputs/local/c45-complete-parent-validation-20261112/frozen-p16-code'
REFERENCE=ROOT/'outputs/research/2026-11-09-silver/extlib'
CLOUD=ROOT/'outputs/forks/c50-full-masked-validation-v1'
EXTRA=CLOUD
C39=ROOT/'outputs/local/c48-auxiliary-holdout-20261112'
POLICY=ROOT/'configs/c49-consensus-pair-20261112.json'


def stamp():return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk:=stream.read(4*1024*1024):digest.update(chunk)
    return digest.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))


def write(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2),encoding='utf8');temporary.replace(path)


def no_outcomes():
    auth=read(ROOT/'.local/c38-active-goal.json');assert auth['active'] and auth['submissions_used']==0
    assert not (OUT/'quality-results.json').exists()
    assert not (ROOT/'outputs/local/c42-fixed-c39-validation-20261111/quality-results.json').exists()
    assert not (ROOT/'outputs/local/c13-confirmation-20261110/evaluation-consumption.json').exists()


def frozen_contract():
    contract=read(CODE/'deployment.json')
    assert sha(ROOT/'kaggle/forks/p16-external-reference-c09/notebook.ipynb')=='d6b9c9625ad8fb7d503341c6fd702bd909101d719ca55e400145fb96f2b2af6a'
    assert sha(CODE/'c09e_policy.json')==contract['policy_sha256']=='ea662270a895dc902cc9dd242afd7cd7f63e4a42ce3468474c6bff73d984043c'
    for name,pin in contract['source_files'].items():assert sha(CODE/name)==pin,name
    return contract


def module(name,filename):
    spec=importlib.util.spec_from_file_location(name,CODE/filename)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


def descriptor(smiles,features):
    key,canonical,formula,_,bits=features.structure(smiles)
    fp=DataStructs.ExplicitBitVect(len(bits));fp.SetBitsFromList(np.flatnonzero(bits).tolist())
    scaffold=MurckoScaffold.MurckoScaffoldSmiles(smiles=canonical,includeChirality=False)
    return dict(key=key,canonical=canonical,formula=formula,scaffold=scaffold,fp=fp)


def chemistry_flags(item,query_keys,query_formulas,query_scaffolds,query_fps,known_official):
    maximum=max(DataStructs.BulkTanimotoSimilarity(item['fp'],query_fps),default=0.)
    return dict(query_identity=item['key'] in query_keys,known_official_alias=item['key'] in known_official,
        same_formula=item['formula'] in query_formulas,same_nonempty_scaffold=bool(item['scaffold'] and item['scaffold'] in query_scaffolds),
        near_query_tanimoto=maximum>=.7,maximum_query_tanimoto=float(maximum))


def precursor_reachability(queries,precursor,threshold):
    """Exact union of the original search windows; no reference outside can act."""
    precursor=np.asarray(precursor,dtype=np.float64);order=np.argsort(precursor,kind='stable')
    sorted_values=precursor[order];reachable=np.zeros(len(precursor),bool)
    for row in queries.itertuples(index=False):
        width=row.precursor_mz*threshold['precursor_ppm']*1e-6+threshold['precursor_absolute_da']
        lo=np.searchsorted(sorted_values,row.precursor_mz-width,side='left')
        hi=np.searchsorted(sorted_values,row.precursor_mz+width,side='right')
        reachable[order[lo:hi]]=True
    return reachable


def mask_c39():
    no_outcomes();pa.set_cpu_count(2);assert rdBase.rdkitVersion=='2026.03.3';RDLogger.DisableLog('rdApp.*')
    source_start=sha(__file__);policy_start=sha(POLICY)
    target=OUT/'auxiliary-masked-reference';assert not target.exists(),'Inspect a partial mask rather than overwrite it'
    write(OUT/'auxiliary-mask-progress.json',dict(checked_utc=stamp(),pid=os.getpid(),stage='verifying_inputs',source_sha256=source_start,quality_metrics_read=False))
    contract=frozen_contract()
    for name,pin in contract['reference_files'].items():assert sha(REFERENCE/name)==pin,name
    collection=read(CLOUD/'collection-audit.json');fidelity=read(CLOUD/'acquisition-fidelity-audit.json')
    assert collection['queries']==35 and collection['forbidden_reference_rows']==0
    assert fidelity['all_original_acquisitions_retained'] and fidelity['acquisitions']==35
    for name,pin in collection['files_sha256'].items():assert sha(CLOUD/name)==pin,name
    freeze=read(C39/'validation-freeze.json');label_path=C39/'labels.csv'
    assert sha(label_path)==freeze['files_sha256'][str(label_path.relative_to(ROOT))]
    labels=pd.read_csv(label_path,dtype=str);assert len(labels)==35
    queries=pd.read_parquet(CLOUD/'c41_queries.parquet');assert len(queries)==35 and queries.molecule_id.nunique()==35
    assert not {'key','smiles','formula','inchikey14'}&set(queries.columns)
    features=module('c45_mask_features','c09e_features.py');signature=module('c45_mask_signatures','c09_spectrum_audit.py').signatures
    described=[]
    for row in labels.itertuples(index=False):
        item=descriptor(row.smiles,features);assert item['key']==row.key and item['formula']==row.formula
        described.append(item)
    qkeys={item['key'] for item in described};qformulas={item['formula'] for item in described}
    qscaffolds={item['scaffold'] for item in described if item['scaffold']};qfps=[item['fp'] for item in described]
    raw_path=ROOT/'outputs/local/c48-clean-residual-heads-20261112/engine2-raw-key-exclusion.json'
    raw_payload=read(raw_path);assert isinstance(raw_payload,dict) and isinstance(raw_payload['keys'],list)
    raw_blocked=set(raw_payload['keys']);assert len(raw_blocked)==32555
    witness_path=ROOT/'outputs/local/c48-clean-residual-heads-20261112/original-structure-identity-witnesses.parquet'
    witnesses=pd.read_parquet(witness_path,columns=['raw_key','official_key'])
    official_blocked=set(witnesses.loc[witnesses.raw_key.isin(raw_blocked),'official_key'].dropna())|qkeys
    query_signatures=set()
    for row in queries.itertuples(index=False):
        values=signature(row,features)
        if values:query_signatures.update(values)
    meta=pd.read_parquet(REFERENCE/'ext_meta.parquet')
    mz=np.load(REFERENCE/'ext_mz.npy',mmap_mode='r');intensity=np.load(REFERENCE/'ext_in.npy',mmap_mode='r')
    assert mz.shape==intensity.shape==(len(meta),128)
    reachable=precursor_reachability(queries,meta.precursor_mz,read(CODE/'c09e_policy.json')['thresholds'])
    reachable_rows=np.flatnonzero(reachable);assert len(reachable_rows)>0
    print('C51 EXACT REACHABLE REFERENCE ROWS',len(reachable_rows),'/',len(meta),flush=True)
    records={};flag_names=['invalid_identity','query_identity','known_official_alias','same_formula','same_nonempty_scaffold','near_query_tanimoto']
    for count,smiles in enumerate(dict.fromkeys(meta.loc[reachable,'normalized_smiles'].astype(str)),1):
        try:
            item=descriptor(smiles,features);flags=chemistry_flags(item,qkeys,qformulas,qscaffolds,qfps,official_blocked)
            records[smiles]=dict(smiles=smiles,official_key=item['key'],formula=item['formula'],scaffold=item['scaffold'],invalid_identity=False,**flags)
        except (ValueError,TypeError,RuntimeError) as error:
            records[smiles]=dict(smiles=smiles,official_key=None,formula=None,scaffold=None,invalid_identity=True,error_type=type(error).__name__,
                **{key:False for key in flag_names if key!='invalid_identity'},maximum_query_tanimoto=None)
        if count%10000==0:
            write(OUT/'auxiliary-mask-progress.json',dict(checked_utc=stamp(),pid=os.getpid(),stage='reference_chemistry',completed_structures=count,quality_metrics_read=False))
            print('C51 REFERENCE STRUCTURES',count,flush=True)
    descriptors=pd.DataFrame(records.values()).set_index('smiles')
    flags=pd.DataFrame(False,index=np.arange(len(meta)),columns=flag_names)
    flags.loc[reachable_rows,flag_names]=descriptors.loc[meta.loc[reachable,'normalized_smiles'].astype(str),flag_names].to_numpy(bool)
    flags['outside_all_frozen_precursor_windows']=~reachable
    flags['known_raw_alias']=meta.inchikey14.astype(str).isin(raw_blocked).to_numpy()
    aliases=np.zeros(len(meta),bool)
    for checked,pos in enumerate(reachable_rows,1):
        values=signature(SimpleNamespace(ms2_mzs=mz[pos],ms2_normalized_intensities=intensity[pos]),features)
        aliases[pos]=bool(values and any(value in query_signatures for value in values))
        if checked%2500==0:
            write(OUT/'auxiliary-mask-progress.json',dict(checked_utc=stamp(),pid=os.getpid(),stage='reference_acquisition_aliases',completed_rows=checked,total_rows=len(reachable_rows),quality_metrics_read=False))
            print('C51 REFERENCE INPUT ALIASES',checked,'/',len(reachable_rows),flush=True)
    flags['query_input_alias']=aliases
    excluded=flags.any(axis=1).to_numpy(bool);keep=np.flatnonzero(~excluded)
    assert len(keep)>0 and excluded.any()
    target.mkdir();descriptors.reset_index().to_parquet(target/'identity-audit.parquet',index=False)
    flags.insert(0,'original_row',np.arange(len(meta)));flags.to_parquet(target/'row-exclusion-audit.parquet',index=False)
    selected=meta.iloc[keep].reset_index(drop=True);selected.to_parquet(target/'ext_meta.parquet',index=False)
    for name,original in [('ext_mz.npy',mz),('ext_in.npy',intensity)]:
        output=np.lib.format.open_memmap(target/name,mode='w+',dtype=original.dtype,shape=(len(keep),128))
        for start in range(0,len(keep),4096):output[start:start+4096]=original[keep[start:start+4096]]
        output.flush();del output
        actual=np.load(target/name,mmap_mode='r')
        for start in range(0,len(keep),4096):assert np.array_equal(actual[start:start+4096],original[keep[start:start+4096]],equal_nan=True)
    pd.testing.assert_frame_equal(pd.read_parquet(target/'ext_meta.parquet'),meta.iloc[keep].reset_index(drop=True),check_exact=True)
    reasons=flag_names+['outside_all_frozen_precursor_windows','known_raw_alias','query_input_alias']
    assert not flags.loc[keep,reasons].any().any()
    assert sha(__file__)==source_start and sha(POLICY)==policy_start,'Mask implementation or protocol changed during preparation'
    report=dict(frozen_utc=stamp(),kind='C48_auxiliary_complete_P16_reference_holdout',source_sha256=source_start,policy_sha256=policy_start,
        original_deployment_sha256=sha(CODE/'deployment.json'),original_reference_files=contract['reference_files'],
        query_sha256=sha(CLOUD/'c41_queries.parquet'),exclusion_structure_source_sha256=sha(label_path),
        original_global_raw_alias_sha256=sha(raw_path),original_structure_witness_sha256=sha(witness_path),
        known_raw_aliases=len(raw_blocked),known_official_aliases=len(official_blocked),queries=35,acquisitions=35,
        original_rows=len(meta),excluded_rows=int(excluded.sum()),retained_rows=len(keep),
        exact_precursor_reachable_rows=len(reachable_rows),lossless_precursor_pruning=True,
        nonexclusive_reasons={key:int(flags[key].sum()) for key in reasons},
        forbidden_rows_retained=0,retained_order_and_all_peak_fields_exact=True,production_reference_changed=False,
        quality_metrics_read=False,C13_read=False,
        limitation='Controlled query-chemistry and bounded signature exclusion; public model pretraining exposure remains disclosed. Rounded full-peak/top96 signatures do not prove universal provenance exclusion.',
        reference_files={name:sha(target/name) for name in contract['reference_files']},
        audit_files={name:sha(target/name) for name in ['identity-audit.parquet','row-exclusion-audit.parquet']})
    write(target/'mask-manifest.json',report)
    write(OUT/'auxiliary-mask-progress.json',dict(checked_utc=stamp(),pid=os.getpid(),stage='complete',retained_rows=len(keep),excluded_rows=int(excluded.sum()),quality_metrics_read=False))
    print('C51 MASK COMPLETE',json.dumps({k:v for k,v in report.items() if k not in ['reference_files','audit_files']}),flush=True)


def independent_overlay(baseline,matches,confidence,policy,identity):
    lookup=matches.set_index('molecule_id').to_dict('index');result=baseline.copy();threshold=policy['thresholds']
    for i,row in baseline.iterrows():
        value=confidence.get(str(row.molecule_id));hit=lookup.get(str(row.molecule_id))
        if value is None or not np.isfinite(value) or value>=threshold['protect_pipeline_library_at']:continue
        if not hit or hit['cosine']<threshold['sqrt_cosine'] or hit['matched_peaks']<threshold['minimum_matched_peaks'] or not hit['smiles']:continue
        try:key=identity(hit['smiles'])
        except (ValueError,TypeError):continue
        strings=row.smiles.split(';');keys=[identity(s) for s in strings]
        if key==keys[0]:continue
        chosen=strings[keys.index(key)] if key in keys else hit['smiles']
        result.at[i,'smiles']=';'.join(([chosen]+[s for s,k in zip(strings,keys) if k!=key])[:25])
    return result


def audit_replay(work):
    work=Path(work);receipt=read(work/'c09e_external_receipt.json');contract=read(work/'validation-deployment.json')
    assert receipt['required_stage_completed'] and receipt['policy_sha256']==frozen_contract()['policy_sha256']
    assert receipt['contract_sha256']==sha(work/'validation-deployment.json') and receipt['reference_files']==contract['reference_files']
    assert receipt['source_files']==contract['source_files']==frozen_contract()['source_files']
    for name,pin in contract['source_files'].items():assert sha(work/name)==pin,name
    before=pd.read_csv(work/'c09e_p01_submission.csv',dtype=str,keep_default_na=False)
    after=pd.read_csv(work/'submission.csv',dtype=str,keep_default_na=False)
    evidence=pd.read_parquet(work/'frozen_candidate_evidence.parquet')
    assert evidence.groupby('molecule_id').library_max.nunique(dropna=False).eq(1).all() and np.isfinite(evidence.library_max).all()
    confidence=evidence.groupby('molecule_id',sort=False).library_max.first().to_dict();assert set(before.molecule_id)<=set(confidence)
    features=module('c45_replay_identity','c09e_features.py')
    @lru_cache(maxsize=50000)
    def identity(smiles):return features.structure(smiles)[0]
    matches=pd.read_parquet(work/'c09e_matches.parquet');actual=matches[matches.view.eq('all_reference')]
    rebuilt=independent_overlay(before,actual,confidence,read(CODE/'c09e_policy.json'),identity)
    pd.testing.assert_frame_equal(rebuilt,after,check_exact=True)
    rank1=membership=changed=0
    for old,new in zip(before.smiles,after.smiles):
        a,b=old.split(';'),new.split(';');ak,bk=[identity(s) for s in a],[identity(s) for s in b]
        assert 1<=len(b)<=25 and len(set(bk))==len(bk)
        rank1+=ak[0]!=bk[0];membership+=set(ak)!=set(bk);changed+=a!=b
    assert changed==receipt['changed_rows']
    result=dict(completed_utc=stamp(),source_sha256=sha(__file__),receipt_sha256=sha(work/'c09e_external_receipt.json'),
        baseline_sha256=sha(work/'c09e_p01_submission.csv'),parent_sha256=sha(work/'submission.csv'),queries=len(before),
        independent_fixed_policy_reconstruction_exact=True,finite_complete_library_confidence=True,
        changed_lists=changed,rank1_changes=rank1,candidate_membership_changes=membership,official_identity_duplicates=0,
        quality_metrics_read=False,C13_read=False)
    write(work/'independent-parent-audit.json',result);print('C51 INDEPENDENT PARENT AUDIT',json.dumps(result),flush=True)
    return result


def replay_c39():
    import shutil
    no_outcomes();contract=frozen_contract();mask=OUT/'auxiliary-masked-reference';manifest=read(mask/'mask-manifest.json')
    assert manifest['policy_sha256']==sha(POLICY) and manifest['source_sha256']==sha(__file__) and manifest['forbidden_rows_retained']==0
    for name,pin in {**manifest['reference_files'],**manifest['audit_files']}.items():assert sha(mask/name)==pin,name
    extra=['frozen_candidate_evidence.parquet','outsider_neural_features.parquet'];work=OUT/'auxiliary-parent';assert not work.exists()
    work.mkdir();shutil.copy2(CLOUD/'c41_p01.csv',work/'submission.csv');shutil.copy2(CLOUD/'c41_queries.parquet',work/'effective_query_spectra.parquet')
    for name in list(contract['source_files'])+['c09e_policy.json']:shutil.copy2(CODE/name,work/name)
    shutil.copy2(CLOUD/'c41_neural_p01.parquet',work/'neural_candidate_features.parquet')
    for name in extra:shutil.copy2(EXTRA/name,work/name)
    contract=copy.deepcopy(contract);contract.update(reference_files=manifest['reference_files'],validation_only=True,
        reference_mask_manifest_sha256=sha(mask/'mask-manifest.json'),original_deployment_sha256=sha(CODE/'deployment.json'))
    write(work/'validation-deployment.json',contract)
    with (work/'frozen-P16-replay.log').open('w',encoding='utf8') as stream:
        result=subprocess.run([sys.executable,str(work/'apply_external_reference.py'),'--work',str(work),'--input-root',str(mask),'--contract',str(work/'validation-deployment.json')],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
    assert result.returncode==0,'Inspect the frozen replay log; do not weaken its contract'
    audit=audit_replay(work);receipt=read(work/'c09e_external_receipt.json')
    assert receipt['search']['queries']==35 and receipt['search']['spectra']==35 and receipt['search']['query_input_alias_reference_rows']==0
    for output,source in [('P01.csv',CLOUD/'c41_p01.csv'),('P14.csv',CLOUD/'c41_p14.csv'),('queries.parquet',CLOUD/'c41_queries.parquet'),('P16.csv',work/'submission.csv')]:
        target=OUT/output;assert not target.exists();shutil.copy2(source,target)
    write(OUT/'complete-parent-freeze.json',dict(frozen_utc=stamp(),source_sha256=sha(__file__),policy_sha256=sha(POLICY),
        mask_manifest_sha256=sha(mask/'mask-manifest.json'),parent_audit_sha256=sha(work/'independent-parent-audit.json'),
        files_sha256={name:sha(OUT/name) for name in ['P01.csv','P14.csv','P16.csv','queries.parquet']},
        candidate_family_sha256=sha(OUT/'candidate-family-freeze.json'),quality_metrics_read=False,C13_read=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['mask','replay','audit']);parser.add_argument('--work',type=Path);args=parser.parse_args()
    if args.phase=='mask':mask_c39()
    elif args.phase=='replay':replay_c39()
    else:assert args.work is not None;audit_replay(args.work)
