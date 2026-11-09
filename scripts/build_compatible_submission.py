"""P15: complete pinned P01 followed by the selected two-fit C08 tail adjustment."""
import hashlib,json,re
from pathlib import Path
from build_general_anchor import ROOT,replace_function,guarded_source


def substitute(source,**values):
    pattern=r'\b('+'|'.join(re.escape(key) for key in values)+r')\b'
    return re.sub(pattern,lambda match:repr(values[match.group(0)]),source)


def main():
    bundle=ROOT/'outputs/publish/c08-models-20261108'
    manifest=bundle/'c08-manifest.json';selection=bundle/'selection.json'
    m=json.loads(manifest.read_text());s=json.loads(selection.read_text())
    assert not m['smoke'] and m['steps']==40000 and len(m['models'])==8
    pin=hashlib.sha256(manifest.read_bytes()).hexdigest();selection_pin=hashlib.sha256(selection.read_bytes()).hexdigest()
    assert s['manifest_sha256']==pin and s['explicit_user_trial']
    winner=s['winner'];assert winner['weight'] in [.05,.1,.2]
    nb=json.loads((ROOT/'kaggle/forks/p01-public480/notebook.ipynb').read_text(encoding='utf-8'))
    swap='''def _swap_to_ours(base_bundle):
    _pin='c08d2101154d057890de088be63834eb622468892a3b842512fdc5a65e02ff46'
    _hits=[p for p in Path('/kaggle/input').rglob('final_model.pt')
           if 'casmi_spectrum_model_ours_v80' in str(p) and hashlib.sha256(p.read_bytes()).hexdigest()==_pin]
    assert len(_hits)==1,'Exactly one pinned P01 v80 checkpoint required'
    _dst=ROOT/'ours_bundle';_dst.mkdir(exist_ok=True)
    for _name in ['model.py','features.py','config.json']:shutil.copy2(Path(base_bundle)/_name,_dst/_name)
    shutil.copy2(_hits[0],_dst/'final_model.pt')
    return str(_dst)'''
    replace_function(nb,'_swap_to_ours',swap)
    preflight='''
# A bundle-layout fallback must never silently swap the champion's public model.
for _public_name,_public_pin in MODEL_PINS.items():
    assert hashlib.sha256((Path(BUNDLE)/_public_name).read_bytes()).hexdigest()==_public_pin,_public_name
_c08_hits=list(Path('/kaggle/input').rglob('c08-manifest.json'))
assert len(_c08_hits)==1 and hashlib.sha256(_c08_hits[0].read_bytes()).hexdigest()==MODEL_PIN
C08_BUNDLE=_c08_hits[0].parent
assert hashlib.sha256((C08_BUNDLE/'selection.json').read_bytes()).hexdigest()==SELECTION_PIN
C08_MANIFEST=json.loads(_c08_hits[0].read_text())
C08_SELECTION=json.loads((C08_BUNDLE/'selection.json').read_text())
assert C08_MANIFEST['steps']==40000 and not C08_MANIFEST['smoke']
for _name,_digest in C08_MANIFEST['files'].items():
    assert hashlib.sha256((C08_BUNDLE/_name).read_bytes()).hexdigest()==_digest,_name
assert C08_SELECTION['winner']['arm']==ARM and C08_SELECTION['winner']['weight']==WEIGHT
(ROOT/'c08_model_audit.json').write_text(json.dumps(dict(manifest_sha256=MODEL_PIN,selection_sha256=SELECTION_PIN,
    arm=ARM,weight=WEIGHT,models=2,folds=[0,1],steps=40000,smoke=False,
    integration='tail-only RRF after complete P01',baseline_v80_sha256='c08d2101154d057890de088be63834eb622468892a3b842512fdc5a65e02ff46'),indent=2))
'''
    preflight=substitute(preflight,MODEL_PIN=pin,SELECTION_PIN=selection_pin,ARM=winner['arm'],WEIGHT=winner['weight'])
    count=0
    for cell in nb['cells']:
        if cell['cell_type']!='code':continue
        source=''.join(cell['source']);needle='BUNDLE = materialise_sources()'
        if needle in source:source=source.replace(needle,needle+'\n'+preflight);count+=1
        cell['source']=source.splitlines(keepends=True)
    assert count==1
    tail='''# The complete P01 file is the protected baseline for the learned tail stage.
shutil.copy2(ROOT/'submission.csv',ROOT/'c08_p01_submission.csv')
run_stage([sys.executable,C08_BUNDLE/'compatible_inference.py','--bundle',C08_BUNDLE,
    '--arm',ARM,'--weight',str(WEIGHT),'--baseline',ROOT/'c08_p01_submission.csv',
    '--queries',ROOT/'effective_query_spectra.parquet','--scores',ROOT/'c08_candidate_scores.parquet',
    '--output',ROOT/'submission.csv'],ROOT/'c08_inference.log','Required C08 tail scoring')
_before=pd.read_csv(ROOT/'c08_p01_submission.csv',dtype=str,keep_default_na=False)
_after=pd.read_csv(ROOT/'submission.csv',dtype=str,keep_default_na=False)
assert len(_before)==len(_after)==len(SAMPLE_IDS) and _after.molecule_id.tolist()==_before.molecule_id.tolist()
for _b,_a in zip(_before.smiles,_after.smiles):
    _bv,_av=_b.split(';'),_a.split(';')
    assert _av[0]==_bv[0] and len(_av)==len(_bv) and set(_av)==set(_bv)
_receipt=json.loads((ROOT/'submission.csv.c08.json').read_text())
assert _receipt['manifest_sha256']==MODEL_PIN and not _receipt['smoke'] and _receipt['steps']==40000
assert _receipt['input_errors']==0 and _receipt['models']==2 and _receipt['folds']==[0,1]
assert _receipt['rank_one_changes']==_receipt['candidate_set_changes']==0
if not IS_RERUN:assert _receipt['scored_molecules']==12
_scores=pd.read_parquet(ROOT/'c08_candidate_scores.parquet');_error=0.
for _mid,_g in _scores.groupby('molecule_id',sort=False):
    if not _g.c08_ok.all():continue
    _expected=0.
    for _f in [0,1]:
        _v=_g['c08_raw_fold'+str(_f)].to_numpy(dtype=float)
        _expected=_expected+.5*(_v-_v.mean())/max(float(_v.std()),1e-6)
    _error=max(_error,float(np.abs(_g.c08_score.to_numpy()-_expected).max()))
assert _error<1e-6
_receipt.update(model=json.loads((ROOT/'c08_model_audit.json').read_text()),twofit_standardization_error=_error,
    pipeline=json.loads((ROOT/'execution_receipt.json').read_text()),merge_status=MERGE_REPORT['status'],is_hidden_rerun=IS_RERUN)
(ROOT/'p15_qualification_receipt.json').write_text(json.dumps(_receipt,indent=2))
print('P15 QUALIFIED',json.dumps(_receipt),flush=True)
'''
    tail=substitute(tail,MODEL_PIN=pin,ARM=winner['arm'],WEIGHT=winner['weight'])
    nb['cells'].append(dict(cell_type='code',execution_count=None,metadata={},outputs=[],source=tail.splitlines(keepends=True)))
    for i,cell in enumerate(nb['cells']):
        if cell['cell_type']=='code':
            source=''.join(cell['source']);compile(source,'p15','exec');cell['source']=guarded_source(source,f'p15_{i}').splitlines(keepends=True)
            cell['outputs']=[];cell['execution_count']=None
        cell['id']=hashlib.sha256((str(i)+''.join(cell['source'])).encode()).hexdigest()[:12]
    nb['cells'].insert(0,dict(cell_type='markdown',id='p15-header',metadata={},source=[
        '# P15: P01 with the C08 learned tail\n',
        f"Two40000-update320-wide fits, arm {winner['arm']}, tail-RRF weight {winner['weight']}.\n",
        'The complete P01 first choice and candidate membership are protected. No model-weight averaging.\n',
        'Development-only choice on64 novel external full-pipeline queries;64 reserved queries are reported separately.\n',
        'One user-requested experiment; hidden-score improvement and statistical promotion remain unproven.\n',
        'Saner architecture/features, Deotte common-fold workflow and V7 recipe inspiration; upstream credits below are retained.\n']))
    nb['nbformat_minor']=5
    out=ROOT/'kaggle/forks/p15-compatible-c08';out.mkdir(parents=True,exist_ok=True)
    (out/'notebook.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
    meta=json.loads((ROOT/'kaggle/forks/p01-public480/kernel-metadata.json').read_text())
    meta.update(id='clarkkitchen/casmi26-p15-compatible-c08',title='casmi26 p15 compatible c08',is_private=True)
    meta['dataset_sources'].append('clarkkitchen/casmi26-c08-models-20261108')
    (out/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    (out/'model-pin.json').write_text(json.dumps(dict(manifest_sha256=pin,selection_sha256=selection_pin,arm=winner['arm'],weight=winner['weight'],
        models=2,steps=40000,rank_one_protected=True,candidate_set_protected=True,scoring_submission=False),indent=2))
    print('PACKAGED P15; real Kaggle smoke required',out,flush=True)


if __name__=='__main__':main()
