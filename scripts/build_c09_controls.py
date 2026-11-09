"""Build private dual-device C09 P14/coverage controls from pinned C08 code."""
import ast,hashlib,json
from pathlib import Path
from build_general_anchor import ROOT,guarded_source


def unwrap(source):
    tree=ast.parse(source)
    if len(tree.body)==1 and isinstance(tree.body[0],ast.Try):
        node=tree.body[0].body[0]
        if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and getattr(node.value.func,'id',None)=='exec':
            return ast.literal_eval(node.value.args[0].args[0])
    return source


def build():
    parent=ROOT/'kaggle/forks/c08-external-validation/notebook.ipynb'
    nb=json.loads(parent.read_text(encoding='utf-8'))
    frozen_pin=hashlib.sha256((ROOT/'outputs/publish/frozen-p14-20261107/frozen-manifest.json').read_bytes()).hexdigest()
    scorer=(ROOT/'scripts/score_frozen_channel.py').read_text(encoding='utf-8')
    counts=dict(preflight=0,scoring=0)
    for i,cell in enumerate(nb['cells']):
        if cell['cell_type']!='code':continue
        s=unwrap(''.join(cell['source']))
        trigger='BUNDLE = materialise_sources()'
        if trigger in s:
            setup="\n_c09_hits=list(INPUT.rglob('frozen-manifest.json'))\nassert len(_c09_hits)==1\nC09_FROZEN_BUNDLE=_c09_hits[0].parent\nassert hashlib.sha256(_c09_hits[0].read_bytes()).hexdigest()=="+repr(frozen_pin)+"\n(CODE/'score_frozen_channel.py').write_text("+repr(scorer)+",encoding='utf-8')\n"
            s=s.replace(trigger,trigger+setup);counts['preflight']+=1
        trigger='OUTSIDERS_OK = score_outside_candidates(needed=True)'
        if trigger in s:
            code="shutil.copy2(ROOT/'neural_candidate_features.parquet',ROOT/'neural_p01.parquet')\nrun_stage([sys.executable,CODE/'score_frozen_channel.py',C09_FROZEN_BUNDLE,PUB_BUNDLE,ROOT/'neural_p01.parquet',ROOT/'effective_query_spectra.parquet',ROOT/'neural_candidate_features.parquet','cuda'],ROOT/'c09_p14.log','Exact failed P14 negative control')\n"
            s=s.replace(trigger,code+'\n'+trigger);counts['scoring']+=1
        # The original validation seal evaluates the final output only after
        # all baseline, P14, outsider, physical, prior and dedup paths complete.
        s=s.replace('c08_validation_p01_metrics.csv','c09_p14_metrics.csv').replace('c08_validation_p01.csv','c09_p14.csv')
        s=s.replace('c08_validation_report.json','c09_p14_report.json').replace('c08_validation_final_evidence.parquet','c09_p14_final_evidence.parquet')
        cell['source']=guarded_source(s,f'c09_control_{i}').splitlines(keepends=True);cell['outputs']=[];cell['execution_count']=None
    assert counts==dict(preflight=1,scoring=1),counts
    modules={name:(ROOT/'scripts'/name).read_text(encoding='utf-8') for name in ['c09_coverage.py','compatible_inference.py','compatible_tail.py']}
    preamble="""import os,sys,json,subprocess
from pathlib import Path
import torch
_c09_devices=[dict(index=i,name=torch.cuda.get_device_name(i),vram_gib=torch.cuda.get_device_properties(i).total_memory/2**30) for i in range(torch.cuda.device_count())]
assert _c09_devices,'GPU allocation required'
Path('/kaggle/working/c09_devices.json').write_text(json.dumps(_c09_devices,indent=2))
print('C09 ACTUAL DEVICES',json.dumps(_c09_devices),flush=True)
_c09_code=Path('/kaggle/working/c09_code');_c09_code.mkdir(exist_ok=True)
for _name,_source in C09_MODULES.items():(_c09_code/_name).write_text(_source,encoding='utf-8')
_c09_env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='1' if len(_c09_devices)>1 else '0')
_c09_command=[sys.executable,str(_c09_code/'c09_coverage.py'),'/kaggle/input','/kaggle/working/c09_expanded']
_c09_log=open('/kaggle/working/c09_expanded.log','w')
_c09_proc=subprocess.Popen(_c09_command,env=_c09_env,stdout=_c09_log,stderr=subprocess.STDOUT) if len(_c09_devices)>1 else None
os.environ['CUDA_VISIBLE_DEVICES']='0'
""".replace('C09_MODULES',repr(modules))
    nb['cells'].insert(0,dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=preamble.splitlines(keepends=True),id='c09-devices'))
    ending="""# Join the complementary lane and compare the exact failed P14 control.
if _c09_proc is None:_c09_proc=subprocess.Popen(_c09_command,env=_c09_env,stdout=_c09_log,stderr=subprocess.STDOUT)
assert _c09_proc.wait(timeout=18000)==0,'C09 expanded-candidate lane failed; inspect its log'
_c09_log.close()
_c09_input=list(Path('/kaggle/input').rglob('c08_validation_p01_metrics.csv'))
assert len(_c09_input)==1
_before=pd.read_csv(_c09_input[0]).set_index('molecule_id')
_after=pd.read_csv(ROOT/'c09_p14_metrics.csv').set_index('molecule_id').reindex(_before.index)
assert len(_before)==128 and not _after.mrr25.isna().any()
_diff=(_after.mrr25-_before.mrr25).to_numpy()
_rng=np.random.default_rng(20261109);_boot=np.array([_rng.choice(_diff,len(_diff),replace=True).mean() for _ in range(10000)])
_report=dict(kind='full_pipeline_failed_P14_negative_control',queries=128,baseline_mrr25=float(_before.mrr25.mean()),p14_mrr25=float(_after.mrr25.mean()),delta=float(_diff.mean()),ci95=np.quantile(_boot,[.025,.975]).tolist(),improved=int((_diff>0).sum()),harmed=int((_diff<0).sum()),known_public_p01=.480,known_public_p14=.458,detects_known_regression=bool(_diff.mean()<0),limitation='Previously used external panel; source-shift diagnostic, not an untouched holdout')
(ROOT/'c09_negative_control.json').write_text(json.dumps(_report,indent=2))
shutil.copy2(ROOT/'submission.csv',ROOT/'c09_p14_diagnostic.csv')
(ROOT/'submission.csv').unlink()
print('DONE C09 NEGATIVE CONTROL',json.dumps(_report),flush=True)
"""
    nb['cells'].append(dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=ending.splitlines(keepends=True),id='c09-comparison'))
    for i,c in enumerate(nb['cells']):
        if c['cell_type']=='code':compile(''.join(c['source']),f'c09_{i}','exec')
    out=ROOT/'kaggle/forks/c09-coverage-controls';out.mkdir(parents=True,exist_ok=True)
    (out/'notebook.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
    meta=json.loads(parent.with_name('kernel-metadata.json').read_text())
    meta.update(id='clarkkitchen/casmi26-c09-coverage-controls',title='casmi26 c09 coverage controls',is_private=True)
    meta['dataset_sources']+=['clarkkitchen/casmi26-frozen-p14-20261107','clarkkitchen/casmi26-c08-models-20261108']
    meta['kernel_sources'].append('clarkkitchen/casmi26-c08-novel-external-validation')
    (out/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    pin=dict(campaign='C09',scoring_submission=False,parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),p14_manifest_sha256=frozen_pin,
        notebook_sha256=hashlib.sha256((out/'notebook.ipynb').read_bytes()).hexdigest(),device_contract='GPU0 exact P14 full pipeline; GPU1 C08 expanded actual pools; serial fallback on one GPU')
    (out/'control-pin.json').write_text(json.dumps(pin,indent=2),encoding='utf-8');print('BUILT C09 CONTROLS',json.dumps(pin))


if __name__=='__main__':build()
