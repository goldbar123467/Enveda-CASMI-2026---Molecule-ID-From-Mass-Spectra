"""Package only completed final C28 folds. No upload or quality selection."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import importlib.util
import json
import shutil
from datetime import datetime,timezone

import torch
from precision_cache import ROOT,sha,write

RUN=ROOT/'outputs/local/c28-multiview-20261110'
PACKAGE=ROOT/'outputs/publish/c28-multiview-models-20261110'
EVALUATION=ROOT/'configs/c28-evaluation-20261110.json'


def package():
    torch.set_num_threads(2)
    contract=json.loads((RUN/'source-contract.json').read_text())
    for path,pin in contract['source_sha256'].items():assert sha(ROOT/path)==pin,path
    assert contract['spec']['training']['steps']==5000 and contract['spec']['folds']==[0,1]
    assert contract['spec']['run_id']=='C28MV'
    results=json.loads((RUN/'results.json').read_text())
    assert [r['fold'] for r in results['models']]==[0,1]
    existing=PACKAGE/'c08-manifest.json'
    if existing.exists():
        manifest=json.loads(existing.read_text())
        assert manifest['run_id']=='C28MV' and manifest['training_contract_sha256']==sha(RUN/'source-contract.json')
        assert manifest['evaluation_plan_sha256']==sha(EVALUATION) and manifest['packager_sha256']==sha(__file__)
        for name,pin in manifest['files'].items():assert sha(PACKAGE/name)==pin,name
        return manifest
    parent=ROOT/'outputs/publish/c08-models-20261108'
    base=json.loads((parent/'c08-manifest.json').read_text())
    parents={m['fold']:m for m in base['models'] if m['arm']=='legacy16-hard'}
    assert set(parents)=={0,1}
    PACKAGE.mkdir(exist_ok=True);models=[];receipts=[]
    for fold in [0,1]:
        folder=RUN/'models/multi-view'/('fold-'+str(fold));path=folder/'final_model.pt'
        receipt=json.loads((folder/'complete.json').read_text())
        assert receipt==results['models'][fold]
        assert receipt['continuation_steps']==5000 and receipt['parent_steps']==40000
        assert receipt['attempts']==5000+receipt['amp_skips'] and receipt['amp_skips']/receipt['attempts']<.02
        assert receipt['heldout_in_training']==0 and receipt['parent_sha256']==parents[fold]['checkpoint_sha256']
        assert receipt['checkpoint_sha256']==sha(path)
        counts=receipt['view_sampling_counts']
        assert counts['primary_queries']==receipt['attempts']*contract['spec']['training']['batch_size']
        assert counts['two_view_queries']+counts['single_view_queries']==counts['primary_queries']
        assert counts['two_view_queries']>0
        ck=torch.load(path,map_location='cpu',weights_only=True)
        assert ck['fold']==fold and ck['steps']==45000 and ck['continuation_steps']==5000 and not ck['smoke']
        assert ck['parent_sha256']==receipt['parent_sha256']
        assert ck['cfg']['name']=='multi-view' and ck['cfg']['cache_precision']=='float16'
        assert all(ck['cfg'][k]==v for k,v in base['architecture'].items())
        assert all(torch.isfinite(v).all() for v in ck['state_dict'].values())
        filename='multi-view-fold'+str(fold)+'.pt';shutil.copy2(path,PACKAGE/filename)
        models.append(dict(file=filename,fold=fold,arm='multi-view',steps=45000,cache_precision='float16',checkpoint_sha256=sha(path)))
        receipts.append(receipt);del ck
    names=['model.py','features.py','fingerprint-cache.npy','fingerprint-cache-smiles.parquet','compatible_tail.py']
    for name in names:
        assert sha(parent/name)==base['files'][name],name
        shutil.copy2(parent/name,PACKAGE/name)
    source=(ROOT/'scripts/compatible_inference.py').read_text(encoding='utf-8')
    needle='expected_steps=200 if smoke else 40000';assert source.count(needle)==1
    source=source.replace(needle,"expected_steps=45000\n    assert not smoke and manifest['run_id']=='C28MV'")
    compile(source,'c28_pinned_inference','exec')
    (PACKAGE/'inference.py').write_text(source,encoding='utf-8',newline='\n')
    names+=['inference.py']+[m['file'] for m in models]
    manifest=dict(kind='compatible_scorer',run_id='C28MV',steps=45000,smoke=False,
        architecture=base['architecture'],models=models,files={n:sha(PACKAGE/n) for n in names},
        training_contract_sha256=sha(RUN/'source-contract.json'),evaluation_plan_sha256=sha(EVALUATION),
        packager_sha256=sha(__file__),parent_manifest_sha256=sha(parent/'c08-manifest.json'),
        automatic_submission=False,selection='Final5000additional updates/fold only; one fixed all-mode0.20tail recipe.',
        attribution='Credited Saner/P01 compatible model/features; original local C08 parents. Original terms retained.')
    write(PACKAGE/'artifact-audit.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),
        models=2,all_finite=True,checkpoint_receipts=receipts,private_upload_required=True,
        inference_adaptation='Only final cumulative-step/run-id acceptance changes; scoring arithmetic unchanged.',
        quality_gate_passed=False,automatic_submission=False))
    write(existing,manifest)
    write(PACKAGE/'dataset-metadata.json',dict(id='clarkkitchen/casmi26-c28-multiview-models-20261110',
        title='CASMI26 C28 same-polarity multiview models',licenses=[dict(name='CC-BY-NC-4.0')]))
    (PACKAGE/'README.md').write_text('Private C28MV final same-polarity multiview models, based on the credited Saner/P01 model and local C08 parents. See immutable training/evaluation contract hashes and original terms. Packaging asserts no quality gain. No automatic upload or submission.\n',encoding='utf-8')
    print('DONE C28 FINAL ARTIFACT AUDIT',sha(existing),flush=True)
    return manifest


def serving_module():
    manifest=package()
    spec=importlib.util.spec_from_file_location('_c28_pinned_serving',PACKAGE/'inference.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return manifest,module


if __name__=='__main__':package()
