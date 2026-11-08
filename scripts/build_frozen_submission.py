"""P14: P01 plus development-selected F07R1 final-list channel; mandatory smoke."""
import hashlib,json
from pathlib import Path
from build_general_anchor import ROOT,replace_function,guarded_source

DATASET='clarkkitchen/casmi26-frozen-p14-20261107'
V80='c08d2101154d057890de088be63834eb622468892a3b842512fdc5a65e02ff46'


def build():
    folder=ROOT/'outputs/publish/frozen-p14-20261107'
    pin=hashlib.sha256((folder/'frozen-manifest.json').read_bytes()).hexdigest()
    report=json.loads((ROOT/'outputs/local/long-frozen-20261107-r1/qualification/report.json').read_text())
    assert report['positive_confirmation'] and report['confirmation']['ci95'][0]>0
    stable=json.loads((ROOT/'outputs/local/long-frozen-20261107-r1/qualification/stable_sort_report.json').read_text())
    assert stable['positive_confirmation'] and not stable['reselected']
    nb=json.loads((ROOT/'kaggle/forks/p01-public480/notebook.ipynb').read_text(encoding='utf-8'))
    swap='''def _swap_to_ours(base_bundle):
    _hits=[p for p in Path('/kaggle/input').rglob('final_model.pt')
           if 'casmi_spectrum_model_ours_v80' in str(p) and hashlib.sha256(p.read_bytes()).hexdigest()==V80PIN]
    assert len(_hits)==1,'Exactly one pinned P01 v80 scorer is required'
    _dst=ROOT/'ours_bundle';_dst.mkdir(exist_ok=True)
    for _name in ['model.py','features.py','config.json']:shutil.copy2(Path(base_bundle)/_name,_dst/_name)
    shutil.copy2(_hits[0],_dst/'final_model.pt')
    return str(_dst)'''.replace('V80PIN',repr(V80))
    replace_function(nb,'_swap_to_ours',swap)
    scorer='''def score_final_lists():
    run_stage([sys.executable,CODE/'neural_scores.py',BUNDLE,'final_model.pt',
               ROOT/'frozen_candidate_evidence.parquet',ROOT/'effective_query_spectra.parquet',
               ROOT/'neural_candidate_features.parquet','cuda'],ROOT/'submission_neural.log','Required P01 final scorer')
    return True'''
    replace_function(nb,'score_final_lists',scorer)
    driver=(ROOT/'scripts/score_frozen_channel.py').read_text(encoding='utf-8');added=0;patched=0;preflight=0
    for cell in nb['cells']:
        if cell['cell_type']!='code':continue
        s=''.join(cell['source'])
        if "MODULES['neural_scores.py']" in s:
            s+='\nMODULES["score_frozen_channel.py"]='+repr(driver)+'\nMODULE_SHA256["score_frozen_channel.py"]='+repr(hashlib.sha256(driver.encode()).hexdigest())+'\n';added+=1
        trigger='BUNDLE = materialise_sources()'
        if trigger in s:
            code='''
_hits=list(Path('/kaggle/input').rglob('frozen-manifest.json'))
assert len(_hits)==1,'Exactly one P14 frozen dataset is required'
assert hashlib.sha256(_hits[0].read_bytes()).hexdigest()==FROZENPIN
FROZEN_BUNDLE=_hits[0].parent
_manifest=json.loads(_hits[0].read_text())
for _name,_digest in _manifest['files'].items():
    assert hashlib.sha256((FROZEN_BUNDLE/_name).read_bytes()).hexdigest()==_digest,_name
import torch as _torch,gc as _gc
_graph=_torch.jit.load(str(FROZEN_BUNDLE/'dreams.ts'),map_location='cuda').eval()
_x=_torch.zeros((2,101,2),device='cuda');_x[:,0,0]=300;_x[:,0,1]=1.1;_x[:,1,0]=100;_x[:,1,1]=1
with _torch.inference_mode():assert _torch.isfinite(_graph(_x)).all()
del _graph,_x;_gc.collect();_torch.cuda.empty_cache()
_graph=_torch.jit.load(str(FROZEN_BUNDLE/'molformer.ts'),map_location='cuda').eval()
_ids=_torch.full((2,202),2,device='cuda',dtype=_torch.long);_ids[:,0]=0;_ids[:,1]=1
_mask=_torch.zeros_like(_ids);_mask[:,:2]=1
with _torch.inference_mode():assert _torch.isfinite(_graph(_ids,_mask)).all()
del _graph,_ids,_mask;_gc.collect();_torch.cuda.empty_cache()
(ROOT/'p14_model_audit.json').write_text(json.dumps(dict(manifest_sha256=FROZENPIN,head_sha256=_manifest['files']['head.pt'],
    selected_seed=_manifest['selected_seed'],weight=_manifest['fusion_weight'],v80_sha256=V80PIN),indent=2))
print('P14 OFFLINE ENCODER PREFLIGHT PASSED',flush=True)
'''.replace('FROZENPIN',repr(pin)).replace('V80PIN',repr(V80))
            s=s.replace(trigger,trigger+'\n'+code);preflight+=1
        trigger='OUTSIDERS_OK = score_outside_candidates(needed=True)'
        if trigger in s:
            code='''shutil.copy2(ROOT/'neural_candidate_features.parquet',ROOT/'neural_p01.parquet')
run_stage([sys.executable,CODE/'score_frozen_channel.py',FROZEN_BUNDLE,PUB_BUNDLE,
           ROOT/'neural_p01.parquet',ROOT/'effective_query_spectra.parquet',
           ROOT/'neural_candidate_features.parquet','cuda'],ROOT/'p14_frozen.log','Required F07R1 final-list channel')
'''
            s=s.replace(trigger,code+'\n'+trigger);patched+=1
        cell['source']=s.splitlines(keepends=True)
    assert added==patched==preflight==1
    seal='''_base=pd.read_parquet(ROOT/'neural_p01.parquet')
_final=pd.read_parquet(ROOT/'neural_candidate_features.parquet')
assert len(_base)==len(_final) and _base[['molecule_id','smiles']].equals(_final[['molecule_id','smiles']])
_receipt=json.loads((ROOT/'neural_candidate_features.parquet.frozen.json').read_text())
_weight=float(_receipt['weight'])
assert _receipt['manifest_sha256']==FROZENPIN
assert MERGE_REPORT.get('status')=='applied'
if not IS_RERUN:
    assert _receipt['scored_molecules']>0 and _receipt['input_errors']==0,'Real supported inputs required for smoke'
_error=0.
for _mid,_g in _final.groupby('molecule_id',sort=False):
    _ix=_g.index;_old=_base.loc[_ix,'neural_score'].to_numpy(dtype=float)
    if _g.frozen_ok.astype(bool).all():
        _v=_g.frozen_score.to_numpy(dtype=float);_znew=(_v-_v.mean())/max(float(_v.std()),1e-6)
        _expect=(1-_weight)*_old+_weight*(_znew*max(float(_old.std()),1e-6)+_old.mean())
    else:_expect=_old
    _error=max(_error,float(np.abs(_g.neural_score.to_numpy()-_expect).max()))
assert _error<1e-6
_receipt.update(model=json.loads((ROOT/'p14_model_audit.json').read_text()),frozen_fusion_max_abs_error=_error,
    real_scored_molecules=int(_base.loc[_base.neural_ok.astype(bool),'molecule_id'].nunique()),
    merge_status=MERGE_REPORT['status'],pipeline=json.loads((ROOT/'execution_receipt.json').read_text()),is_hidden_rerun=IS_RERUN)
(ROOT/'p14_qualification_receipt.json').write_text(json.dumps(_receipt,indent=2))
print('P14 QUALIFIED',json.dumps(_receipt),flush=True)
'''.replace('FROZENPIN',repr(pin))
    nb['cells'].append(dict(cell_type='code',execution_count=None,metadata={},outputs=[],source=seal.splitlines(keepends=True)))
    for i,c in enumerate(nb['cells']):
        if c['cell_type']=='code':
            s=''.join(c['source']);compile(s,'p14','exec');c['source']=guarded_source(s,f'p14_{i}').splitlines(keepends=True);c['outputs']=[];c['execution_count']=None
        c.setdefault('id',hashlib.sha256((str(i)+''.join(c['source'])).encode()).hexdigest()[:12])
    nb['cells'].insert(0,dict(cell_type='markdown',id='p14-header',metadata={},source=[
        '# P14: P01 plus F07R1 frozen DreaMS/MoLFormer final-list channel\n',
        'Seed202611071, development-selected50% complementary neural channel; original P01 outsider/physical/prior/dedup recipe.\n',
        'Fresh256 reserved conditional-ranking MRR .637481 to .742269; truth-forced candidates, not end-to-end or Kaggle MRR.\n',
        'Official encoder provenance, licenses and exact checkpoint hashes are retained. Unsupported spectra retain P01.\n']))
    nb['nbformat_minor']=5
    out=ROOT/'kaggle/forks/p14-frozen-channel';out.mkdir(parents=True,exist_ok=True)
    (out/'notebook.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
    meta=json.loads((ROOT/'kaggle/forks/p01-public480/kernel-metadata.json').read_text());meta.update(id='clarkkitchen/casmi26-p14-frozen-channel',title='casmi26 p14 frozen channel',is_private=True)
    meta['dataset_sources'].append(DATASET)
    (out/'kernel-metadata.json').write_text(json.dumps(meta,indent=2));(out/'model-pin.json').write_text(json.dumps(dict(manifest_sha256=pin,dataset=DATASET,v80_sha256=V80,selection=report['winner']),indent=2))
    print('PACKAGED P14',pin)


if __name__=='__main__':build()
