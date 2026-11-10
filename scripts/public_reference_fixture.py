"""Controlled real-input C33 overlay audit with independent peak assignment."""
import ast
import hashlib
import json
import textwrap
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from public_deployment_support import intake,protect,finalize
from c29_reference_audit import reference_view,signatures


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv_sha(frame):return hashlib.sha256(frame.to_csv(index=False,lineterminator='\n').encode()).hexdigest()


def clean(mz,intensity,precursor=None):
    mz,it=np.asarray(mz,np.float64),np.asarray(intensity,np.float64)
    keep=(mz>0)&(it>0)
    if precursor is not None:keep&=mz<=precursor+.5
    mz,it=mz[keep],it[keep];ix=np.argsort(mz);return mz[ix],it[ix]


def vector_match(qm,qi,rm,ri):
    if not len(qm) or not len(rm):return 0.,0
    a,b=np.sqrt(qi),np.sqrt(ri);a/=np.linalg.norm(a)+1e-12;b/=np.linalg.norm(b)+1e-12
    x,y=np.nonzero(np.abs(qm[:,None]-rm[None,:])<=.015)
    products=a[x]*b[y];order=np.lexsort((-y,-x,-products))
    used_q=np.zeros(len(qm),bool);used_r=np.zeros(len(rm),bool);score=0.;count=0
    for i in order:
        if not used_q[x[i]] and not used_r[y[i]]:
            used_q[x[i]]=True;used_r[y[i]]=True;score+=float(products[i]);count+=1
    return score,count


def independent(queries,parent,confidence,input_root,features,public_match):
    ext=list(input_root.rglob('ext_meta.parquet'));massbank=[p for p in input_root.rglob('spectra.parquet') if 'massbank' in str(p).lower()]
    assert len(ext)==len(massbank)==1
    meta=pd.read_parquet(ext[0]);mass=pd.read_parquet(ext[0].with_name('ext_mass.parquet'))
    mz=np.load(ext[0].with_name('ext_mz.npy'),mmap_mode='r');it=np.load(ext[0].with_name('ext_in.npy'),mmap_mode='r')
    mb=pd.read_parquet(massbank[0],columns=['smiles','exact_mass','ion_mode','mzs','intensities','is_casmi_subset'])
    mb=mb[(~mb.is_casmi_subset.astype(bool))&mb.smiles.notna()].reset_index(drop=True)
    if 'exact_mass' in mass.columns:
        assert len(meta)==len(mass)
        valid=mass.exact_mass.notna()&mass.exact_mass.gt(0)
    else:
        # The published v2 file is a compound-level mono_mass table. The
        # production recipe leaves spectrum metadata unfiltered in that case.
        valid=pd.Series(True,index=meta.index)
    entries=[(float(meta.precursor_mz.iloc[i]),0,int(i)) for i in np.flatnonzero(valid.to_numpy())]
    entries += [(float(r.exact_mass)+(1.007276 if 'pos' in str(r.ion_mode).lower() else -1.007276),1,i) for i,r in enumerate(mb.itertuples(index=False))]
    order=np.argsort([r[0] for r in entries],kind='stable');entries=[entries[i] for i in order];prec=np.array([r[0] for r in entries])
    prepared={};sig=set()
    for row in queries.itertuples(index=False):
        sig.update(signatures(row.ms2_mzs,row.ms2_normalized_intensities,features))
        qm,qi=clean(row.ms2_mzs,row.ms2_normalized_intensities,float(row.precursor_mz));sig.update(signatures(qm,qi,features))
    winners={str(mid):{v:(0.,0,None) for v in ['all_reference','input_alias_excluded']} for mid in parent.molecule_id}
    error=0.;pairs=0
    for row in queries.itertuples(index=False):
        mid=str(row.molecule_id);qprec=float(row.precursor_mz);mode=1 if 'pos' in str(row.ionization_mode).lower() else -1
        qm,qi=clean(row.ms2_mzs,row.ms2_normalized_intensities,qprec);width=qprec*10e-6+.002
        lo=np.searchsorted(prec,qprec-width,'left');hi=np.searchsorted(prec,qprec+width,'right')
        for index in range(lo,hi):
            _,source,pos=entries[index];tag=(source,pos)
            if tag not in prepared:
                if source==0:
                    r=meta.iloc[pos];rm,ri=clean(mz[pos],it[pos]);rmode=1 if 'pos' in str(r.get('ionization_mode','pos')).lower() else -1;smiles=str(r.normalized_smiles)
                else:
                    r=mb.iloc[pos];rm,ri=clean(r.mzs,r.intensities);rmode=1 if 'pos' in str(r.ion_mode).lower() else -1;smiles=str(r.smiles)
                prepared[tag]=(rm,ri,rmode,smiles,bool(set(signatures(rm,ri,features))&sig))
            rm,ri,rmode,smiles,alias=prepared[tag]
            if rmode!=mode or len(rm)<3:continue
            score,n=vector_match(qm,qi,rm,ri);other,k=public_match(qm,qi,rm,ri)
            error=max(error,abs(score-other));assert n==k and error<1e-10
            pairs+=1
            if n<4:continue
            for view in winners[mid]:
                if view=='input_alias_excluded' and alias:continue
                if score>winners[mid][view][0]:winners[mid][view]=(score,n,smiles)
    identity={}
    def key(s):
        if s not in identity:identity[s]=features.structure(s)[0]
        return identity[s]
    results={}
    for view in ['all_reference','input_alias_excluded']:
        rows=[]
        for row in parent.itertuples(index=False):
            original=row.smiles.split(';');score,n,hit=winners[str(row.molecule_id)][view]
            if confidence.get(str(row.molecule_id),0.)>=.90 or hit is None or score<.80:proposed=original
            elif hit in original:proposed=[hit]+[s for s in original if s!=hit]
            else:proposed=([hit]+original)[:25]
            seen=set();answer=[]
            for s in proposed+original:
                k=key(s)
                if k not in seen:answer.append(s);seen.add(k)
                if len(answer)==25:break
            rows.append(dict(molecule_id=row.molecule_id,smiles=';'.join(answer)))
        results[view]=pd.DataFrame(rows)
    return results,dict(peak_pairs=pairs,peak_score_max_abs_difference=error,reference_candidate_rows=len(prepared),
        sourceB_CASMI_subset_excluded=True,matching_implementation='Vectorized distance matrix and lexsort/boolean masks; production uses searchsorted/pair tuples/sets.')


def run(work,input_root,package,overlay_source,features,known_adducts,expected=None):
    work=Path(work);input_root=Path(input_root);package=Path(package);work.mkdir(parents=True,exist_ok=True)
    contract=json.loads((package/'c09e-fixture-contract.json').read_text());assert contract['queries']==30 and not contract['query_truth_labels_included']
    for name,pin in contract['files'].items():assert sha(package/name)==pin,name
    parent=pd.read_csv(package/'c09e_fixture_baseline.csv',dtype=str,keep_default_na=False)
    queries=pd.read_parquet(package/'c09e_fixture_queries.parquet');assert queries.molecule_id.nunique()==30
    confidence=pd.read_csv(package/'c09e_fixture_library.csv').set_index('molecule_id').library_max.to_dict()
    first=queries.iloc[0].to_dict();sample=parent.smiles.iloc[0]
    extra=[dict(molecule_id='C33F_'+x,smiles=sample) for x in ['absent','unknown','adduct','short']]
    parent=pd.concat([parent,pd.DataFrame(extra)],ignore_index=True)
    add=[dict(first,molecule_id='C33F_unknown',ionization_mode='?'),dict(first,molecule_id='C33F_adduct',adduct='UNKNOWN'),
        dict(first,molecule_id='C33F_short',ms2_mzs=np.array([50.,100.]),ms2_normalized_intensities=np.array([1.,.5]))]
    queries=pd.concat([queries,pd.DataFrame(add)],ignore_index=True)
    eligible,intake_audit=intake(parent,queries,features,known_adducts);assert len(eligible)==30
    q=queries[queries.molecule_id.astype(str).isin(eligible)].copy();query_path=work/'eligible-queries.parquet';q.to_parquet(query_path,index=False)
    scope={'C33_OVERLAY_QUERY_PATH':query_path,'C33_ELIGIBLE_COUNT':30,'_c29_features':features,'reference_view':reference_view}
    exec(compile(overlay_source,'c33_reviewed_overlay_fixture','exec'),scope)
    outer=ast.parse(overlay_source).body[0]
    node=next(n for n in ast.walk(outer) if isinstance(n,ast.FunctionDef) and n.name=='calc_cosine')
    numeric={'np':np};exec(compile(textwrap.dedent(ast.get_source_segment(overlay_source,node)),'c33_numeric_primitive','exec'),numeric)
    wanted,agreement=independent(q,parent,confidence,input_root,features,numeric['calc_cosine'])
    identity={}
    def key(s):
        if s not in identity:identity[s]=features.structure(s)[0]
        return identity[s]
    views={};outputs={}
    for view,exclude in [('all_reference',False),('input_alias_excluded',True)]:
        target=work/view;target.mkdir(exist_ok=True);parent.to_csv(target/'submission.csv',index=False)
        pd.DataFrame(dict(molecule_id=list(confidence),library_max=list(confidence.values()))).to_parquet(target/'frozen_candidate_evidence.parquet',index=False)
        scope['_c29_exclude']=exclude;scope['_apex_extlib_overlay'](work=str(target),inp=str(input_root))
        raw=pd.read_csv(target/'submission.csv',dtype=str,keep_default_na=False)
        final,identity_report=finalize(raw,parent,key,parent);final=protect(parent,final,eligible)
        # The independent path derives both winners and ordering separately.
        independent_final=protect(parent,wanted[view],eligible)
        assert final.equals(independent_final),'Independent public overlay/order disagrees'
        guard=~parent.molecule_id.isin(eligible);assert final.loc[guard].equals(parent.loc[guard])
        final.to_csv(target/'final.csv',index=False)
        audit=json.loads((target/'apex_extlib_audit.json').read_text());assert audit['sourceA_valid_rows']>0 and audit['sourceB_valid_rows']>0
        changed=int(final.smiles.ne(parent.smiles).sum());assert changed>0
        views[view]=dict(predictions_content_sha256=csv_sha(final),changed_rows=changed,production_audit=audit,identity_report=identity_report)
        outputs[view]=final
    result=dict(kind='C33_real_input_reference_fixture',real_queries=30,synthetic_fallback_cases=4,unsupported_parent_rows_exact=4,
        fixture_contract_sha256=sha(package/'c09e-fixture-contract.json'),overlay_source_sha256=hashlib.sha256(overlay_source.encode()).hexdigest(),
        helper_sha256=sha(__file__),views=views,independent=agreement,query_truth_labels_used=False,quality_metrics_computed=False)
    if expected is not None:
        assert result['overlay_source_sha256']==expected['overlay_source_sha256']
        assert result['fixture_contract_sha256']==expected['fixture_contract_sha256']
        for view in views:
            assert views[view]['predictions_content_sha256']==expected['views'][view]['predictions_content_sha256']
            assert views[view]['changed_rows']==expected['views'][view]['changed_rows']
    (work/'fixture-receipt.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result
