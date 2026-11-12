"""Build an explicitly audited single-2070 port and a USED engineering fixture.

No new validation labels/outcomes are read. Every scientific cell remains exact;
the sole inner-code adaptation maps the original Engine2recovery GPU1 to GPU0.
The complete original outer public/ours neural blend and ranking remain present.
"""
import os
for n in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(n,'2')
import torch
import argparse,ast,base64,copy,hashlib,json,shutil
from pathlib import Path
import pandas as pd
from c75_port_utils import ROOT,read,write,sha,stamp,unwrap,value,replace_value
OUT=ROOT/'outputs/local/c75-local-core-port-20261112'


def draw(mid):return hashlib.sha256(('C75-USED12-port-parity-v1:'+mid).encode()).hexdigest()


def hardware_cell(original):
    tree=ast.parse(original);preflight=None
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='write_text' and node.args:
            try:text=ast.literal_eval(node.args[0])
            except (ValueError,TypeError):continue
            if isinstance(text,str) and 'P14 GPU1 REAL ENCODER PREFLIGHT' in text:preflight=text
    assert preflight is not None
    preflight=preflight.replace('P14 GPU1 REAL ENCODER PREFLIGHT','P14 LOCAL GPU0 REAL ENCODER PREFLIGHT')
    return '''import os,json,sys,subprocess,torch
from pathlib import Path
_c75_devices=[dict(index=i,name=torch.cuda.get_device_name(i),vram_gib=torch.cuda.get_device_properties(i).total_memory/2**30) for i in range(torch.cuda.device_count())]
assert len(_c75_devices)==1 and '2070' in _c75_devices[0]['name'],'Actual local RTX2070required'
Path('/kaggle/working/c75_gpu_allocation.json').write_text(json.dumps(dict(devices=_c75_devices,local_single_GPU=True,actual_two_T4s=False,allocation={'GPU0':['original complete P01pipeline','original real P14encoder preflight','original P14negative control']}),indent=2))
Path('/kaggle/working/c41_gpu1_preflight.py').write_text('''+repr(preflight)+''',encoding='utf-8')
_c41t4_log=open('/kaggle/working/c41_gpu1_preflight.log','w')
_c41t4_child=subprocess.Popen([sys.executable,'/kaggle/working/c41_gpu1_preflight.py'],env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),stdout=_c41t4_log,stderr=subprocess.STDOUT)
os.environ['CUDA_VISIBLE_DEVICES']='0'
print('C75 ACTUAL LOCAL DEVICE',json.dumps(_c75_devices),flush=True)
'''


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['fixture','full']);args=parser.parse_args();mode=args.mode
    target=OUT/(mode+'-port');target.mkdir(exist_ok=True);assert not (target/'port-freeze.json').exists()
    assert not (ROOT/'outputs/local/c72-clean-target-holdout-20261112/evaluation-consumption.json').exists()
    assert not (ROOT/'outputs/local/c13-confirmation-20261110/evaluation-consumption.json').exists()
    if mode=='full':assert read(OUT/'fixture-parity-audit.json')['passed']
    prefix='c57' if mode=='fixture' else 'c73'
    source=ROOT/f'kaggle/forks/{prefix}-full-masked-validation/notebook.ipynb'
    nb=read(source);original=copy.deepcopy(nb)
    sources={i:unwrap(''.join(c['source'])) for i,c in enumerate(nb['cells']) if c['cell_type']=='code'}
    package=target/'query-package';package.mkdir(exist_ok=True)
    old_package=ROOT/f'outputs/publish/{prefix}-full-masked-validation-20261111'
    contract=read(old_package/'validation-contract.json');old_contract_sha=sha(old_package/'validation-contract.json')
    for name,pin in contract['files'].items():assert sha(old_package/name)==pin,name
    q=pd.read_parquet(old_package/'validation-queries.parquet')
    if mode=='fixture':
        groups=dict(tuple(q.groupby('molecule_id',sort=False)));modes={mid:'mixed' if g.ionization_mode.nunique()>1 else str(g.ionization_mode.iloc[0]) for mid,g in groups.items()}
        selected=[]
        for query_mode in ['positive','mixed','negative']:
            candidates=[mid for mid in groups if modes[mid]==query_mode]
            ordered=sorted(candidates,key=lambda mid:(not bool(groups[mid].adduct.eq('[M+H]+').any()),draw(mid)))
            assert len(ordered)>=3;selected+=ordered[:3]
        selected+=sorted(set(groups)-set(selected),key=draw)[:3];assert len(set(selected))==12
        q=q[q.molecule_id.isin(selected)].copy()
        selection=dict(method='Metadata-only deterministic hash;3peractualmode,preferH+ to exerciseGL;3remaining hashed. No prediction or quality selection.',
            queries=12,acquisitions=len(q),actual_modes=pd.Series([modes[mid] for mid in selected]).value_counts().to_dict(),
            original_C57_cloud_queries=482,all_selected_original_acquisitions_retained=True,ids=selected,USED_only=True)
        q.to_parquet(package/'validation-queries.parquet',index=False)
        ids=list(dict.fromkeys(q.molecule_id));pd.DataFrame(dict(molecule_id=ids,smiles='CCO')).to_csv(package/'validation-sample.csv',index=False)
        for name in ['main-reference-sid-exclusion.npy','engine2-raw-key-exclusion.json']:shutil.copyfile(old_package/name,package/name)
        (package/'ATTRIBUTION.txt').write_text('USED12query engineering parity fixture, all original C53/C57acquisitions and ORIGINAL completedC57reference masks. No source labels or predictions mounted. All original public dataset attribution retained. No fresh quality or scoring claim.\n',encoding='utf8')
        contract.update(queries=12,spectra=len(q),mode='USED_real12_local_port_parity',quality_metrics_computed=False)
        contract['files']={n:sha(package/n) for n in contract['files']}
        write(package/'validation-contract.json',contract);write(target/'fixture-selection.json',selection)
    else:
        for name in list(contract['files'])+['validation-contract.json']:shutil.copyfile(old_package/name,package/name)
        assert q.molecule_id.nunique()==512 and len(q)==2215
    count=q.molecule_id.nunique();new_contract_sha=sha(package/'validation-contract.json')
    outer=sources[12];payload=base64.b64decode(value(outer,'FROZEN_B64'));original_inner=read_inner=json.loads(payload)
    assert hashlib.sha256(payload).hexdigest()==value(outer,'FROZEN_SHA256')=='3f973abbb1593e356db4f6dec149c8ad99f5afa5ded8da8bab59152bd49e60e7'
    inner=copy.deepcopy(original_inner);before=''.join(inner['cells'][43]['source']);needle="'CUDA_VISIBLE_DEVICES':'1'";assert before.count(needle)==1
    after=before.replace(needle,"'CUDA_VISIBLE_DEVICES':'0'");inner['cells'][43]['source']=after.splitlines(keepends=True)
    assert after.replace("'CUDA_VISIBLE_DEVICES':'0'",needle)==before
    assert [i for i,(a,b) in enumerate(zip(original_inner['cells'],inner['cells'])) if a['source']!=b['source']]==[43]
    local_payload=json.dumps(inner,ensure_ascii=False,indent=1).encode('utf8');local_sha=hashlib.sha256(local_payload).hexdigest()
    outer=replace_value(outer,'FROZEN_B64',base64.b64encode(local_payload).decode());outer=replace_value(outer,'FROZEN_SHA256',local_sha)
    sources[12]=outer;sources[0]=hardware_cell(sources[0])
    if mode=='fixture':
        for i in [6,27]:
            assert sources[i].count(old_contract_sha)==1
            sources[i]=sources[i].replace(old_contract_sha,new_contract_sha).replace('==482','==12').replace('queries=482','queries=12')
    assert sources[27].count("os.environ['CUDA_VISIBLE_DEVICES']='1'")==1
    sources[27]=sources[27].replace("os.environ['CUDA_VISIBLE_DEVICES']='1'","os.environ['CUDA_VISIBLE_DEVICES']='0'")
    for i,c in enumerate(nb['cells']):
        if c['cell_type']=='code':
            compile(sources[i],'c75_'+str(i),'exec');c['source']=sources[i].splitlines(keepends=True);c['outputs']=[];c['execution_count']=None
    changed=[i for i,s in sources.items() if s!=unwrap(''.join(original['cells'][i]['source']))]
    assert set(changed)==({0,6,12,27} if mode=='fixture' else {0,12,27}),changed
    write(target/'notebook.ipynb',nb);(target/'original-inner.ipynb').write_bytes(payload);(target/'local-inner.ipynb').write_bytes(local_payload)
    write(target/'port-freeze.json',dict(prepared_utc=stamp(),source_sha256=sha(__file__),mode=mode,queries=int(count),acquisitions=len(q),
        original_notebook_sha256=sha(source),original_notebook=str(source.relative_to(ROOT)),original_inner_sha256=hashlib.sha256(payload).hexdigest(),
        local_inner_sha256=local_sha,local_notebook_sha256=sha(target/'notebook.ipynb'),original_masks_preserved=mode=='fixture',
        single_inner_change='Only Engine2recovery CUDA_VISIBLE_DEVICES1->0 in cell43; every other original scientific cell exact',
        original_outer_ensemble_and_ranking_unchanged=True,changed_outer_cells=changed,query_contract_sha256=new_contract_sha,
        adapted_outer_purposes={'0':'Actual single2070allocation and original encoder preflight onGPU0','6':'Fixturequery count/hash only' if mode=='fixture' else 'unchanged',
            '12':'Repackage the single-GPU inner payload with its honest newhash','27':'GPU0negativecontrol; fixture count/hash only'},
        original_outer_cell_sha256={str(i):hashlib.sha256(unwrap(''.join(original['cells'][i]['source'])).encode()).hexdigest() for i in sources},
        local_outer_cell_sha256={str(i):hashlib.sha256(s.encode()).hexdigest() for i,s in sources.items()},
        labels_mounted=False,new_source_quality_read=False,C13_read=False,scoring_entries=0,parity_not_yet_established=True))
    print('C75 PORT/QUERY PACKAGE FROZEN',mode,count,len(q),'queries/acquisitions; parity still required',flush=True)


if __name__=='__main__':main()
