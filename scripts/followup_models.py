"""Pinned serving bundles for completed C09 control/mining milestones."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
import torch
import numpy as np
import pandas as pd
import importlib.util,json,shutil
from pathlib import Path
from precision_cache import ROOT,sha,write

PACKAGE=ROOT/'outputs/publish/c09-followup-models-20261109'
CANDIDATES=['cosine-control','mined-5000','mined-10000','mined-15000','mined-20000']
REFERENCES=['p15-parent','c09-served']


def package():
    torch.set_num_threads(2);out=PACKAGE;out.mkdir(parents=True,exist_ok=True)
    parent=ROOT/'outputs/publish/c08-models-20261108';base=json.loads((parent/'c08-manifest.json').read_text())
    groups=[];audits=[];source_contracts={}
    definitions=[('p15-parent','legacy16-hard','outputs/local/long-compatible-20261108',40000,0),
        ('c09-served','served-score','outputs/local/c09-20261109',45000,5000),
        ('cosine-control','cosine-control','outputs/local/c09-control-20261109',45000,5000)]
    definitions += [(f'mined-{step}','mined128-served','outputs/local/c09-mined-20261109',40000+step,step) for step in [5000,10000,15000,20000]]
    for name,arm,run_path,steps,additional in definitions:
        run=ROOT/run_path
        if additional:
            contract=json.loads((run/'source-contract.json').read_text())
            assert all(sha(ROOT/p)==pin for p,pin in contract['source_sha256'].items())
            source_contracts[run_path]=contract
        models=[]
        for fold in [0,1]:
            folder=run/'models'/arm/f'fold-{fold}'
            path=folder/(f'milestone-{additional}.pt' if name.startswith('mined-') else 'final_model.pt')
            value=torch.load(path,map_location='cpu',weights_only=True)
            assert value['steps']==steps and value['fold']==fold and not value['smoke'] and value['cfg']['name']==arm
            assert value['cfg']['cache_precision']=='float16'
            assert all(value['cfg'][k]==v for k,v in base['architecture'].items())
            assert all(torch.isfinite(v).all() for v in value['state_dict'].values())
            receipt=json.loads((folder/'complete.json').read_text())
            if additional:
                assert value['continuation_steps']==additional and receipt['heldout_in_training']==0
                assert receipt['attempts']==receipt['continuation_steps']+receipt['amp_skips']
                assert receipt['amp_skips']/receipt['attempts']<.02
                expected=next(m['checkpoint_sha256'] for m in base['models'] if m['arm']=='legacy16-hard' and m['fold']==fold)
                assert value['parent_sha256']==expected
            if name=='mined-20000':
                final=torch.load(folder/'final_model.pt',map_location='cpu',weights_only=True)
                assert all(torch.equal(final['state_dict'][k],v) for k,v in value['state_dict'].items());del final
            file=f'{name}-fold{fold}.pt';shutil.copy2(path,out/file)
            models.append(dict(file=file,fold=fold,arm=arm,cache_precision='float16',steps=steps,checkpoint_sha256=sha(out/file)))
            audits.append(dict(group=name,fold=fold,cumulative_steps=steps,additional_steps=additional,checkpoint_sha256=sha(path),all_finite=True,
                completed_fit_amp_skips=receipt.get('amp_skips',0),completed_fit_attempts=receipt.get('attempts')))
            del value
        groups.append(dict(name=name,arm=arm,steps=steps,additional_steps=additional,models=models))
    shared=['model.py','features.py','fingerprint-cache.npy','fingerprint-cache-smiles.parquet','compatible_tail.py']
    for name in shared:
        assert sha(parent/name)==base['files'][name];shutil.copy2(parent/name,out/name)
    source=(ROOT/'scripts/compatible_inference.py').read_text(encoding='utf-8')
    needle='expected_steps=200 if smoke else 40000';assert source.count(needle)==1
    source=source.replace(needle,"expected_steps=manifest['steps']\n    assert not smoke and manifest['run_id']=='C09F' and expected_steps in [40000,45000,50000,55000,60000]")
    (out/'inference.py').write_text(source,encoding='utf-8')
    names=shared+['inference.py']+[m['file'] for g in groups for m in g['models']]
    manifest=dict(run_id='C09F',architecture=base['architecture'],groups=groups,files={n:sha(out/n) for n in names},
        candidate_order=CANDIDATES,references=REFERENCES,source_contracts=source_contracts,
        parent_manifest_sha256=sha(parent/'c08-manifest.json'),selection='Development-only; previously used acquisition-clean30 queries; freeze before new80 evaluation',
        causal_comparison='Control and first mined cycle share C09T data/RNG/schedule/microbatch; later mined cycles are dose exploration',automatic_submission=False)
    write(out/'followup-manifest.json',manifest)
    write(out/'artifact-audit.json',dict(models=len(audits),groups=len(groups),all_finite=True,checkpoint_audits=audits,
        note='Mining fits completed full20000 successful updates per fold;5 AMP-skipped attempts per fold are retained, not counted as updates'))
    write(out/'dataset-metadata.json',dict(id='clarkkitchen/casmi26-c09-followup-models-20261109',title='CASMI26 C09 followup models 20261109',licenses=[dict(name='CC-BY-NC-4.0')]))
    print('DONE FOLLOWUP ARTIFACT AUDIT',json.dumps(dict(groups=len(groups),model_files=len(audits),all_finite=True,manifest_sha256=sha(out/'followup-manifest.json'))),flush=True)
    return manifest


def serving_bundle(group_name,cache_root):
    manifest=json.loads((PACKAGE/'followup-manifest.json').read_text());group=next(g for g in manifest['groups'] if g['name']==group_name)
    out=Path(cache_root)/group_name;out.mkdir(parents=True,exist_ok=True)
    names=['model.py','features.py','fingerprint-cache.npy','fingerprint-cache-smiles.parquet','compatible_tail.py','inference.py']+[m['file'] for m in group['models']]
    for name in names:
        source=PACKAGE/name;assert sha(source)==manifest['files'][name]
        if not (out/name).exists():
            try:(out/name).hardlink_to(source)
            except OSError:shutil.copy2(source,out/name)
        assert sha(out/name)==manifest['files'][name]
    write(out/'c08-manifest.json',dict(kind='compatible_scorer',run_id='C09F',steps=group['steps'],smoke=False,
        architecture=manifest['architecture'],models=group['models'],files={n:manifest['files'][n] for n in names}))
    spec=importlib.util.spec_from_file_location('_c09_followup_serving',out/'inference.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return out,group,module


def score_group(group_name,baseline,queries,output,shared_fps,cache_root):
    out,group,module=serving_bundle(group_name,cache_root)
    receipt_path=Path(str(output)+'.json')
    if Path(output).exists() and receipt_path.exists():
        import hashlib
        receipt=json.loads(receipt_path.read_text())
        assert receipt['manifest_sha256']==sha(out/'c08-manifest.json') and receipt['queries_sha256']==sha(queries)
        assert receipt['baseline_content_sha256']==hashlib.sha256(baseline.to_csv(index=False,lineterminator='\n').encode()).hexdigest()
        return pd.read_parquet(output),receipt
    return module.score(out,group['arm'],baseline,queries,output,shared_fps=shared_fps)


if __name__=='__main__':package()
