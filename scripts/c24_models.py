"""Package the two final C24PL checkpoints without uploading or fitting."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import importlib.util
import json
import shutil
from datetime import datetime,timezone
import torch
from precision_cache import ROOT,sha,write

RUN=ROOT/'outputs/local/c24-polarity-loss-20261110'
PACKAGE=ROOT/'outputs/publish/c24-polarity-models-20261110'
EVALUATION=ROOT/'configs/c24-evaluation-20261110.json'


def package():
    torch.set_num_threads(2)
    contract=json.loads((RUN/'source-contract.json').read_text())
    for path,pin in contract['source_sha256'].items():assert sha(ROOT/path)==pin,path
    assert contract['spec']['training']['steps']==5000 and contract['spec']['folds']==[0,1]
    results=json.loads((RUN/'results.json').read_text());assert len(results['models'])==2
    existing=PACKAGE/'c08-manifest.json'
    if existing.exists():
        manifest=json.loads(existing.read_text())
        assert manifest['run_id']=='C24PL' and manifest['training_contract_sha256']==sha(RUN/'source-contract.json')
        assert manifest['evaluation_plan_sha256']==sha(EVALUATION) and manifest['packager_sha256']==sha(__file__)
        for name,pin in manifest['files'].items():assert sha(PACKAGE/name)==pin,name
        return manifest
    parent=ROOT/'outputs/publish/c18-peak-models-20261110'
    architecture_parent=ROOT/'outputs/publish/c08-models-20261108'
    base=json.loads((architecture_parent/'c08-manifest.json').read_text())
    parent_manifest=json.loads((parent/'c08-manifest.json').read_text())
    PACKAGE.mkdir(exist_ok=True);models=[];receipts=[]
    for fold in [0,1]:
        folder=RUN/'models/polarity-loss'/('fold-'+str(fold))
        receipt=json.loads((folder/'complete.json').read_text());path=folder/'final_model.pt'
        audit=json.loads((folder/'parent-state-audit.json').read_text())
        exposure=json.loads((folder/'polarity-exposure-audit.json').read_text())
        entry=next(e for e in parent_manifest['models'] if e['fold']==fold)
        assert sha(parent/entry['file'])==receipt['parent_sha256']==entry['checkpoint_sha256']==audit['parent_checkpoint_sha256']
        assert all(audit[k] for k in ['model_state_exact','optimizer_and_scaler_restored','query_negative_python_torch_cuda_peak_rng_restored','additional_updates_only'])
        assert receipt['continuation_steps']==5000 and receipt['parent_steps']==45000
        assert receipt['attempts']==5000+receipt['amp_skips'] and receipt['amp_skips']/receipt['attempts']<.02
        assert receipt['heldout_in_training']==0 and receipt['checkpoint_sha256']==sha(path)
        ck=torch.load(path,map_location='cpu',weights_only=True)
        assert ck['fold']==fold and ck['steps']==50000 and ck['continuation_steps']==5000 and not ck['smoke']
        assert ck['parent_sha256']==receipt['parent_sha256'] and ck['cfg']['name']=='polarity-loss' and ck['cfg']['cache_precision']=='float16'
        assert ck['cfg']['negative_exposure_probability']==exposure['expected_negative_query_fraction']
        assert all(ck['cfg'][k]==v for k,v in base['architecture'].items())
        assert all(torch.isfinite(v).all() for v in ck['state_dict'].values())
        filename='polarity-loss-fold'+str(fold)+'.pt';shutil.copy2(path,PACKAGE/filename)
        models.append(dict(file=filename,fold=fold,arm='polarity-loss',steps=50000,cache_precision='float16',checkpoint_sha256=sha(path)))
        receipts.append(receipt);del ck
    names=['model.py','features.py','fingerprint-cache.npy','fingerprint-cache-smiles.parquet','compatible_tail.py']
    for name in names:
        assert sha(architecture_parent/name)==base['files'][name],name
        shutil.copy2(architecture_parent/name,PACKAGE/name)
    source=(ROOT/'scripts/compatible_inference.py').read_text(encoding='utf-8')
    needle='expected_steps=200 if smoke else 40000';assert source.count(needle)==1
    source=source.replace(needle,"expected_steps=50000\n    assert not smoke and manifest['run_id']=='C24PL'")
    compile(source,'c24_pinned_inference','exec')
    (PACKAGE/'inference.py').write_text(source,encoding='utf-8',newline='\n')
    names+=['inference.py']+[m['file'] for m in models]
    manifest=dict(kind='compatible_scorer',run_id='C24PL',steps=50000,smoke=False,
        architecture=base['architecture'],models=models,files={n:sha(PACKAGE/n) for n in names},
        training_contract_sha256=sha(RUN/'source-contract.json'),evaluation_plan_sha256=sha(EVALUATION),
        packager_sha256=sha(__file__),parent_manifest_sha256=sha(parent/'c08-manifest.json'),
        architecture_cache_manifest_sha256=sha(architecture_parent/'c08-manifest.json'),automatic_submission=False,
        selection='Only final5000-additional-update models; fixed negative-only tail recipe; no coefficient/checkpoint search',
        attribution='Compatible architecture/features from credited Saner/P01 kit; local C08/C18 parents. Original source/model terms retained.')
    write(PACKAGE/'artifact-audit.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),
        models=2,all_finite=True,checkpoint_receipts=receipts,private_upload_required=True,
        inference_adaptation='Only cumulative-step/run-id acceptance differs from original compatible_inference; scoring arithmetic unchanged',
        quality_gate_passed=False,automatic_submission=False))
    write(existing,manifest)
    write(PACKAGE/'dataset-metadata.json',dict(id='clarkkitchen/casmi26-c24-polarity-models-20261110',
        title='CASMI26 C24 polarity loss models',licenses=[dict(name='CC-BY-NC-4.0')]))
    (PACKAGE/'README.md').write_text('Private C24PL models: two final5000-update polarity-weighted continuations of completed C18 parents. Credited Saner/P01 architecture/features and original terms retained. Packaging does not establish quality or leaderboard gain. No automatic upload or public submission.\n',encoding='utf-8')
    print('DONE C24 FINAL ARTIFACT AUDIT',sha(existing),flush=True)
    return manifest


def serving_module():
    manifest=package()
    spec=importlib.util.spec_from_file_location('_c24_pinned_serving',PACKAGE/'inference.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return manifest,module


if __name__=='__main__':package()
