"""Build a private full P01 pipeline run on the fixed novel external panel.

Changes input paths/preflight provenance only, retaining ranking/model settings.
This is a validation notebook, never a competition scoring submission.
"""
import ast,base64,hashlib,json
from pathlib import Path
from build_general_anchor import ROOT,guarded_source,replace_function


def replace_literal(notebook,name,value):
    count=0
    for cell in notebook['cells']:
        if cell['cell_type']!='code':continue
        source=''.join(cell['source'])
        for node in ast.parse(source).body:
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets):
                before=ast.get_source_segment(source,node);assert source.count(before)==1
                source=source.replace(before,name+' = '+repr(value));count+=1
        cell['source']=source.splitlines(keepends=True)
    assert count==1,(name,count)


def main():
    package=ROOT/'outputs/publish/c08-validation-20261108';contract=package/'validation-contract.json'
    digest=hashlib.sha256(contract.read_bytes()).hexdigest()
    original=ROOT/'kaggle/forks/p01-public480/notebook.ipynb';nb=json.loads(original.read_text(encoding='utf-8'))
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
    frozen=json.loads((ROOT/'outputs/research/c08-pipeline/frozen.ipynb').read_text(encoding='utf-8'))
    c26=''.join(frozen['cells'][26]['source']);needle="COMP = os.path.dirname(find('test.parquet'))"
    assert c26.count(needle)==1
    frozen['cells'][26]['source']=c26.replace(needle,needle+"\nCOMP = os.environ['CASMI_C08_VALIDATION_COMP']").splitlines(keepends=True)
    c28=''.join(frozen['cells'][28]['source']);needle="assert comp in {f'/kaggle/input/{slug}', f'/kaggle/input/competitions/{slug}'}"
    assert c28.count(needle)==1
    frozen['cells'][28]['source']=c28.replace(needle,"assert comp == os.path.realpath(os.environ['CASMI_C08_VALIDATION_COMP'])").splitlines(keepends=True)
    payload=json.dumps(frozen,ensure_ascii=False,indent=1).encode('utf-8');frozen_pin=hashlib.sha256(payload).hexdigest()
    replace_literal(nb,'FROZEN_B64',base64.b64encode(payload).decode());replace_literal(nb,'FROZEN_SHA256',frozen_pin)
    injected=runner_changed=0
    for cell in nb['cells']:
        if cell['cell_type']!='code':continue
        source=''.join(cell['source'])
        needle="_ids = pd.read_parquet(COMP / 'test.parquet', columns=['spectrum_id'])"
        if needle in source:
            setup='''# C08: fixed external queries, no query labels exposed to the pipeline.
_c08_hits=list(INPUT.rglob('validation-contract.json'))
assert len(_c08_hits)==1
assert hashlib.sha256(_c08_hits[0].read_bytes()).hexdigest()==CONTRACT_PIN
C08_VALIDATION_SOURCE=_c08_hits[0].parent
_c08_contract=json.loads(_c08_hits[0].read_text())
assert _c08_contract['queries']==128 and not _c08_contract['truth_injection']
for _name,_pin in _c08_contract['files'].items():
    assert hashlib.sha256((C08_VALIDATION_SOURCE/_name).read_bytes()).hexdigest()==_pin,_name
C08_VALIDATION_COMP=ROOT/'c08_validation_comp';C08_VALIDATION_COMP.mkdir(exist_ok=True)
_training_link=C08_VALIDATION_COMP/'train.parquet'
if not _training_link.exists():_training_link.symlink_to((COMP/'train.parquet').resolve())
shutil.copy2(C08_VALIDATION_SOURCE/'validation-queries.parquet',C08_VALIDATION_COMP/'test.parquet')
shutil.copy2(C08_VALIDATION_SOURCE/'validation-sample.csv',C08_VALIDATION_COMP/'sample_submission.csv')
COMP=C08_VALIDATION_COMP
os.environ['CASMI_C08_VALIDATION_COMP']=str(COMP)
os.environ['CASMI_TEST_PATH']=str(COMP/'test.parquet')
os.environ['CASMI_SAMPLE_PATH']=str(COMP/'sample_submission.csv')
'''.replace('CONTRACT_PIN',repr(digest))
            source=source.replace(needle,setup+'\n'+needle);injected+=1
        if "MODULES['runner.py']" in source or 'MODULES = {' in source:
            # Rewrite the explicit runner only, using the already extracted
            # credited module. Baseline chemistry/ranking are unchanged.
            runner=(ROOT/'outputs/research/c08-pipeline/modules/runner.py').read_text(encoding='utf-8')
            runner=runner.replace("mode='submission',source_adaptations=[]", "mode='external_validation',source_adaptations=['fixed_external_test_path','external_input_preflight']")
            source+='\nMODULES["runner.py"]='+repr(runner)+'\nMODULE_SHA256["runner.py"]='+repr(hashlib.sha256(runner.encode()).hexdigest())+'\n';runner_changed+=1
        cell['source']=source.splitlines(keepends=True)
    assert injected==runner_changed==1,(injected,runner_changed)
    seal='''# Evaluate only AFTER the unchanged P01 pipeline finished generating its answer.
_labels=pd.read_csv(C08_VALIDATION_SOURCE/'labels.csv',dtype={'molecule_id':str,'key':str})
_sub=pd.read_csv(ROOT/'submission.csv',dtype=str,keep_default_na=False)
assert len(_sub)==128 and _sub.molecule_id.tolist()==_labels.molecule_id.tolist()
assert MERGE_REPORT.get('status')=='applied' and IS_RERUN
_queries=pd.read_parquet(ROOT/'effective_query_spectra.parquet')
assert _queries.molecule_id.nunique()==128
assert not {'smiles','normalized_smiles','inchikey','inchikey14','key','formula','compound_name'}&set(_queries.columns)
from rdkit import Chem as _C
from rdkit.Chem.MolStandardize import rdMolStandardize as _S
from functools import lru_cache as _cache
_taut=_S.TautomerEnumerator()
@_cache(maxsize=30000)
def _identity(s):
    _m=_C.MolFromSmiles(s);assert _m is not None
    _C.RemoveStereochemistry(_m);return _C.MolToInchiKey(_taut.Canonicalize(_m))[:14]
_truth=_labels.set_index('molecule_id');_rows=[];_evidence=[]
for _row in _sub.itertuples(index=False):
    _values=_row.smiles.split(';');_keys=[_identity(s) for s in _values]
    assert 1<=len(_keys)<=25 and len(set(_keys))==len(_keys)
    _key=_truth.loc[_row.molecule_id,'key'];_rank=_keys.index(_key)+1 if _key in _keys else None
    _rows.append(dict(molecule_id=_row.molecule_id,partition=_truth.loc[_row.molecule_id,'partition'],truth_present=_rank is not None,
        rank=_rank,mrr25=1/_rank if _rank else 0.,candidate_count=len(_keys)))
    for _rank0,_smiles in enumerate(_values,1):_evidence.append(dict(molecule_id=_row.molecule_id,smiles=_smiles,baseline_rank=_rank0,processed=True))
pd.DataFrame(_rows).to_csv(ROOT/'c08_validation_p01_metrics.csv',index=False)
pd.DataFrame(_evidence).to_parquet(ROOT/'c08_validation_final_evidence.parquet',index=False)
shutil.copy2(ROOT/'submission.csv',ROOT/'c08_validation_p01.csv')
shutil.copy2(ROOT/'effective_query_spectra.parquet',ROOT/'c08_validation_queries.parquet')
_summary=pd.DataFrame(_rows).groupby('partition').agg(queries=('molecule_id','size'),candidate_recall25=('truth_present','mean'),mrr25=('mrr25','mean')).to_dict('index')
_report=dict(kind='full_pipeline_external_validation',queries=128,labels_in_pipeline=False,truth_injection=False,truth_absent_mrr=0,
    validation_contract_sha256=CONTRACT_PIN,cohorts=_summary,
    production_train_structs_sha256=_c08_contract['production_train_structs_sha256'],
    adaptations=['fixed_external_test_path','external_input_preflight'],
    limitation='Novel external identities and source shift; no hidden competition-score guarantee. Broader public foundation/ranker pretraining exposure is not excluded.')
(ROOT/'c08_validation_report.json').write_text(json.dumps(_report,indent=2))
print('DONE C08 FULL PIPELINE VALIDATION',json.dumps(_report),flush=True)
'''.replace('CONTRACT_PIN',repr(digest))
    nb['cells'].append(dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=seal.splitlines(keepends=True)))
    for i,cell in enumerate(nb['cells']):
        if cell['cell_type']=='code':
            source=''.join(cell['source']);compile(source,'c08_validation','exec')
            cell['source']=guarded_source(source,f'c08_validation_{i}').splitlines(keepends=True);cell['outputs']=[];cell['execution_count']=None
        cell['id']=hashlib.sha256((str(i)+''.join(cell['source'])).encode()).hexdigest()[:12]
    nb['nbformat_minor']=5
    out=ROOT/'kaggle/forks/c08-external-validation';out.mkdir(parents=True,exist_ok=True)
    (out/'notebook.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
    meta=json.loads((original.parent/'kernel-metadata.json').read_text())
    meta.update(id='clarkkitchen/casmi26-c08-novel-external-validation',title='casmi26 C08 novel external validation',is_private=True)
    meta['dataset_sources'].append('clarkkitchen/casmi26-c08-validation-20261108')
    (out/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    (out/'validation-pin.json').write_text(json.dumps(dict(parent_p01_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),validation_contract_sha256=digest,
        frozen_sha256=frozen_pin,scoring_submission=False,source_adaptations=['fixed_external_test_path','external_input_preflight']),indent=2),encoding='utf-8')
    print('PACKAGED C08 EXTERNAL VALIDATION',out,flush=True)


if __name__=='__main__':main()
