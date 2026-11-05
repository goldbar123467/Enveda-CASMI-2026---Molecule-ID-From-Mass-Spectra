"""Build P11 from the qualified full-data NP continuation; preserve P08's pipeline."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

from build_general_anchor import ROOT,guarded_source,replace_function


def unwrap_notebook(notebook):
    for cell in notebook['cells']:
        if cell['cell_type']!='code':continue
        tree=ast.parse(''.join(cell['source']))
        assert len(tree.body)==1 and isinstance(tree.body[0],ast.Try)
        call=tree.body[0].body[0].value
        assert isinstance(call,ast.Call) and isinstance(call.func,ast.Name) and call.func.id=='exec'
        compiled=call.args[0]
        assert isinstance(compiled,ast.Call) and isinstance(compiled.func,ast.Name) and compiled.func.id=='compile'
        source=ast.literal_eval(compiled.args[0]);assert isinstance(source,str)
        cell['source']=source.splitlines(keepends=True)


def build(checkpoint,output_dir=None):
    checkpoint=Path(checkpoint)
    assert checkpoint.is_file() and checkpoint.name.startswith('casmi26_seed_23_e32_lr2e-05_')
    assert checkpoint.parent.name=='refit' and checkpoint.parent.parent.name=='np' and checkpoint.parent.parent.parent.name=='production'
    manifest=json.loads((checkpoint.parent.parent/'finetune_manifest.json').read_text())
    assert manifest['mode']=='np' and manifest['stage']=='refit' and not manifest['smoke']
    assert manifest['epochs']==32 and manifest['parent_epoch']==128 and manifest['lr']==2e-5
    assert manifest['mixture_fraction']==.5 and manifest['training_molecules']==273833 and manifest['heldout_identity_sha256'] is None
    parent_pin=json.loads((ROOT/'kaggle/forks/p08-general-anchor/model-pin.json').read_text())['sha256']
    assert manifest['parent_sha256']==parent_pin
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    notebook=json.loads((ROOT/'kaggle/forks/p08-general-anchor/notebook.ipynb').read_text(encoding='utf-8'))
    unwrap_notebook(notebook)
    source='''def _swap_to_ours(base_bundle):
    import hashlib as _h
    import json as _j
    import shutil as _s
    import torch as _t
    from pathlib import Path as _P
    _hits=[p for p in _P('/kaggle/input').rglob(MODEL_NAME)
           if p.parent.name=='refit' and p.parent.parent.name=='np' and p.parent.parent.parent.name=='production']
    assert len(_hits)==1,'Exactly one full-data NP continuation is required'
    _path=_hits[0]
    assert _h.sha256(_path.read_bytes()).hexdigest()==MODEL_SHA
    _manifest=_j.loads((_path.parent.parent/'finetune_manifest.json').read_text())
    assert _manifest['mode']=='np' and _manifest['stage']=='refit' and not _manifest['smoke']
    assert _manifest['epochs']==32 and _manifest['parent_epoch']==128 and _manifest['lr']==2e-5
    assert _manifest['mixture_fraction']==.5 and _manifest['training_molecules']==273833
    assert _manifest['heldout_identity_sha256'] is None and _manifest['parent_sha256']==PARENT_SHA
    assert _manifest['curated_keys_sha256']=='19d59131c440e04b98f43b7307cf53a65d41c2e90680937668170abd3723e420'
    _ck=_t.load(_path,map_location='cpu',weights_only=True)
    assert _ck['stage']=='full_refit' and _ck['epoch']==32 and not _ck.get('smoke',False)
    assert _ck['initialization_parent_epoch']==128 and _ck['continuation_epochs']==32
    assert _ck['initialization_parent_sha256']==PARENT_SHA
    _architecture=dict(width=256,heads=8,layers=4,embedding=192)
    assert _ck['cfg_overrides']==_architecture
    _dst=_P('/kaggle/working/ours_bundle');_dst.mkdir(parents=True,exist_ok=True)
    for _name in ['model.py','features.py']:_s.copy2(_P(base_bundle)/_name,_dst/_name)
    _cfg=_j.loads((_P(base_bundle)/'config.json').read_text());_cfg.update(_architecture)
    (_dst/'config.json').write_text(_j.dumps(_cfg,indent=2));_s.copy2(_path,_dst/'final_model.pt')
    _audit=dict(checkpoint_sha256=MODEL_SHA,stage=_ck['stage'],epoch=32,parent_epoch=128,cumulative_epochs=160,
               parent_sha256=PARENT_SHA,mixture_fraction=.5,architecture=_architecture,
               neural_weights=dict(original=.5,np_curriculum=.5),source=str(_path),training_manifest=_manifest)
    (_P('/kaggle/working')/'curriculum_model_audit.json').write_text(_j.dumps(_audit,indent=2))
    print('NP CURRICULUM PRODUCTION BUNDLE READY',_j.dumps(_audit),flush=True)
    return str(_dst)'''
    source=source.replace('MODEL_NAME',repr(checkpoint.name)).replace('MODEL_SHA',repr(digest)).replace('PARENT_SHA',repr(parent_pin))
    replace_function(notebook,'_swap_to_ours',source)
    for number,cell in enumerate(notebook['cells']):
        if cell['cell_type']!='code':continue
        text=''.join(cell['source']).replace('general20_model_audit.json','curriculum_model_audit.json')
        text=text.replace('p08_qualification_receipt.json','p11_qualification_receipt.json')
        text=text.replace('P08 QUALIFIED','P11 QUALIFIED').replace('P08 AUDIT PASSED: pinned general20 model','P11 AUDIT PASSED: pinned NP curriculum')
        cell['source']=guarded_source(text,f'p11_cell_{number}').splitlines(keepends=True)
    notebook['cells'][0]['source']=[
        '# P11: original 50% + NP curriculum 50%\n',
        'Full-data general20 parent128 epochs +32 fixed NP-mixture epochs atLR2e-5.\n',
        'Both independent excluded-identity parents improve native retrieval and the final panel proxy.\n',
        'Retains P01 documentation weights, physical rules and metric deduplication. Hidden score is unmeasured.\n',
        'Exact checkpoint/parent/source-population guards and numeric ensemble qualification are required.\n']
    folder=Path(output_dir) if output_dir is not None else ROOT/'kaggle/forks/p11-np-anchor'
    folder.mkdir(parents=True,exist_ok=True)
    (folder/'notebook.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding='utf-8')
    meta=json.loads((ROOT/'kaggle/forks/p08-general-anchor/kernel-metadata.json').read_text())
    meta.update(id='clarkkitchen/casmi26-p11-np-anchor',title='casmi26 p11 NP anchor',kernel_sources=['clarkkitchen/casmi26-p10-production-continuation'])
    (folder/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    (folder/'model-pin.json').write_text(json.dumps(dict(file=checkpoint.name,sha256=digest,parent_sha256=parent_pin,stage='full_refit',parent_epochs=128,continuation_epochs=32,mixture=.5,scoring_submission=False),indent=2)+'\n')
    print('PACKAGED P11; real12-molecule GPU smoke required',folder,digest)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('checkpoint',type=Path)
    build(ap.parse_args().checkpoint)
