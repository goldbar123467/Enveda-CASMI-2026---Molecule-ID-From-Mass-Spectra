"""Build P08 after P07's production checkpoint is collected; retain fixed P01 fusion."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def guarded_source(source,label,submission_path='/kaggle/working/submission.csv'):
    # Indenting source would change multiline literal contents and break pins.
    # Execute the unchanged source with the same notebook global namespace.
    wrapped='try:\n    exec(compile('+repr(source)+','+repr(label)+",'exec'),globals())\nexcept BaseException:\n"
    wrapped+='    from pathlib import Path as _required_Path\n    _required_Path('+repr(str(submission_path))+').unlink(missing_ok=True)\n    raise\n'
    compile(wrapped,label+'_guard','exec')
    return wrapped


def replace_function(notebook,name,replacement):
    found=0
    for cell in notebook['cells']:
        if cell['cell_type']!='code':continue
        source=''.join(cell['source'])
        tree=ast.parse(source)
        for node in tree.body:
            if isinstance(node,ast.FunctionDef) and node.name==name:
                original=ast.get_source_segment(source,node)
                assert source.count(original)==1
                source=source.replace(original,replacement)
                found+=1
        cell['source']=source.splitlines(keepends=True)
    assert found==1,(name,found)


def build(checkpoint,output_dir=None):
    checkpoint=Path(checkpoint)
    assert checkpoint.is_file() and checkpoint.name.startswith('casmi26_seed_20_e128_')
    assert checkpoint.parent.name=='refit' and checkpoint.parent.parent.name=='general_full20'
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    notebook=json.loads((ROOT/'kaggle/forks/p01-public480/notebook.ipynb').read_text(encoding='utf-8'))
    replacement='''def _swap_to_ours(base_bundle):
    import hashlib as _h
    import json as _j
    import shutil as _s
    import torch as _t
    from pathlib import Path as _P
    _hits=[p for p in _P('/kaggle/input').rglob(MODEL_NAME)
           if p.parent.name=='refit' and p.parent.parent.name=='general_full20']
    assert len(_hits)==1, 'Exactly one qualified production general20 checkpoint is required'
    _path=_hits[0]
    assert _h.sha256(_path.read_bytes()).hexdigest()==MODEL_SHA256, 'Production checkpoint checksum differs'
    _ck=_t.load(_path,map_location='cpu',weights_only=True)
    assert _ck['stage']=='full_refit' and _ck['epoch']==128
    assert not _ck.get('smoke',False)
    _architecture=dict(width=256,heads=8,layers=4,embedding=192)
    assert _ck['cfg_overrides']==_architecture
    _dst=_P('/kaggle/working/ours_bundle');_dst.mkdir(parents=True,exist_ok=True)
    for _name in ['model.py','features.py']:_s.copy2(_P(base_bundle)/_name,_dst/_name)
    _cfg=_j.loads((_P(base_bundle)/'config.json').read_text())
    _cfg.update(_architecture)
    (_dst/'config.json').write_text(_j.dumps(_cfg,indent=2))
    _s.copy2(_path,_dst/'final_model.pt')
    _audit=dict(checkpoint_sha256=MODEL_SHA256,stage=_ck['stage'],epoch=_ck['epoch'],architecture=_architecture,
               neural_weights=dict(original=.5,general20=.5),source=str(_path))
    (_P('/kaggle/working')/'general20_model_audit.json').write_text(_j.dumps(_audit,indent=2))
    print('GENERAL20 PRODUCTION BUNDLE READY',_j.dumps(_audit),flush=True)
    return str(_dst)'''
    replacement=replacement.replace('MODEL_NAME',repr(checkpoint.name)).replace('MODEL_SHA256',repr(digest))
    replace_function(notebook,'_swap_to_ours',replacement)
    scorer='''def score_final_lists():
    run_stage([sys.executable,CODE/'neural_scores.py',BUNDLE,'final_model.pt',
               ROOT/'frozen_candidate_evidence.parquet',ROOT/'effective_query_spectra.parquet',
               ROOT/'neural_candidate_features.parquet','cuda'],
              ROOT/'submission_neural.log','Required neural scorer on final lists')
    return True'''
    replace_function(notebook,'score_final_lists',scorer)
    seal='''import pandas as _seal_pd
import numpy as _seal_np
_seal_a=_seal_pd.read_parquet(ROOT/'neural_public.parquet').set_index(['molecule_id','smiles'])
_seal_b=_seal_pd.read_parquet(ROOT/'neural_ours.parquet').set_index(['molecule_id','smiles']).reindex(_seal_a.index)
_seal_final=_seal_pd.read_parquet(ROOT/'neural_candidate_features.parquet').set_index(['molecule_id','smiles']).reindex(_seal_a.index)
assert not _seal_a.index.has_duplicates
_seal_eligible=_seal_a.neural_ok.astype(bool)
assert _seal_eligible.any() and _seal_b.loc[_seal_eligible,'neural_ok'].astype(bool).all()
_seal_expected=.5*_z(_seal_a.neural_score.fillna(0))+.5*_z(_seal_b.neural_score.fillna(0))
_seal_error=float((_seal_final.neural_score-_seal_expected).abs().max())
assert _seal_np.isfinite(_seal_final.loc[_seal_eligible,'neural_score']).all()
assert _seal_error<1e-6
assert MERGE_REPORT.get('status')=='applied'
_seal_receipt=dict(model=json.loads((ROOT/'general20_model_audit.json').read_text()),
                  neural_score_max_abs_error=_seal_error,
                  real_scored_molecules=int(_seal_a.loc[_seal_eligible].index.get_level_values(0).nunique()),
                  pipeline=json.loads((ROOT/'execution_receipt.json').read_text()),
                  merge_status=MERGE_REPORT['status'],is_hidden_rerun=IS_RERUN)
(ROOT/'p08_qualification_receipt.json').write_text(json.dumps(_seal_receipt,indent=2))
print('P08 QUALIFIED',json.dumps(_seal_receipt),flush=True)
'''
    notebook['cells'].append(dict(cell_type='code',execution_count=None,metadata={},outputs=[],source=seal.splitlines(keepends=True)))
    # Any extension failure must remove a pre-existing frozen submission, so an
    # exception cannot silently score that unrelated fallback CSV.
    for number,cell in enumerate(notebook['cells']):
        if cell['cell_type']!='code':continue
        source=''.join(cell['source'])
        source=source.replace('REPRODUCTION AUDIT PASSED: public480 assets, ensemble, merge, dedup',
                              'P08 AUDIT PASSED: pinned general20 model, ensemble, merge, dedup')
        assert not any(isinstance(node,ast.ImportFrom) and node.module=='__future__' for node in ast.parse(source).body)
        wrapped=guarded_source(source,f'p08_cell_{number}')
        cell['source']=wrapped.splitlines(keepends=True)
    header=dict(cell_type='markdown',metadata={},source=[
        '# P08: original 50% + general20 wide model 50%\n',
        'Fixed final128-epoch full refit, width256/four layers/eight heads/embedding192.\n',
        'Retains P01 documentation weights, physical rules and metric deduplication.\n',
        'Two independent holdout fits improve the reused physical panel; hidden score is unmeasured.\n',
        'The source notebook and original authors are credited in the retained cells below.\n'])
    notebook['cells'].insert(0,header)
    folder=Path(output_dir) if output_dir is not None else ROOT/'kaggle/forks/p08-general-anchor'
    folder.mkdir(parents=True,exist_ok=True)
    (folder/'notebook.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding='utf-8')
    meta=json.loads((ROOT/'kaggle/forks/p01-public480/kernel-metadata.json').read_text())
    meta.update(id='clarkkitchen/casmi26-p08-general-anchor',title='casmi26 p08 general anchor',kernel_sources=['clarkkitchen/casmi26-p07-refit-v4-replica'])
    (folder/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    (folder/'model-pin.json').write_text(json.dumps(dict(file=checkpoint.name,sha256=digest,stage='full_refit',epochs=128,scoring_submission=False),indent=2)+'\n')
    print('PACKAGED P08; real12-molecule GPU smoke required',folder,digest)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('checkpoint',type=Path)
    build(ap.parse_args().checkpoint)
