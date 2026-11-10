"""Audit/package only the final two completed C18PK fits; never uploads."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import importlib.util
import json
import shutil
from datetime import datetime,timezone

import torch

from precision_cache import ROOT,sha,write

RUN=ROOT/'outputs/local/c18-peak-drop-20261110'
PACKAGE=ROOT/'outputs/publish/c18-peak-models-20261110'
EVALUATION=ROOT/'configs/c18-evaluation-20261110.json'


def package():
    torch.set_num_threads(2)
    contract=json.loads((RUN/'source-contract.json').read_text())
    for path,pin in contract['source_sha256'].items():assert sha(ROOT/path)==pin,path
    assert contract['spec']['training']['steps']==5000 and contract['spec']['folds']==[0,1]
    assert (RUN/'results.json').exists(),'Both final fits must be complete; do not package an intermediate checkpoint'
    existing=PACKAGE/'c08-manifest.json'
    if existing.exists():
        manifest=json.loads(existing.read_text())
        assert manifest['run_id']=='C18PK' and manifest['training_contract_sha256']==sha(RUN/'source-contract.json')
        assert manifest['evaluation_plan_sha256']==sha(EVALUATION)
        for name,pin in manifest['files'].items():assert sha(PACKAGE/name)==pin,name
        return manifest
    parent=ROOT/'outputs/publish/c08-models-20261108'
    base=json.loads((parent/'c08-manifest.json').read_text())
    matched=json.loads((RUN/'matched-control-audit.json').read_text())
    PACKAGE.mkdir(exist_ok=True);models=[];receipts=[]
    for fold in [0,1]:
        folder=RUN/'models/peak-drop'/('fold-'+str(fold))
        receipt=json.loads((folder/'complete.json').read_text());path=folder/'final_model.pt'
        assert receipt['continuation_steps']==5000 and receipt['parent_steps']==40000
        assert receipt['attempts']==5000+receipt['amp_skips'] and receipt['amp_skips']/receipt['attempts']<.02
        assert receipt['heldout_in_training']==0 and receipt['parent_sha256']==matched['parent_sha256'][str(fold)]
        assert receipt['checkpoint_sha256']==sha(path)
        ck=torch.load(path,map_location='cpu',weights_only=True)
        assert ck['fold']==fold and ck['steps']==45000 and ck['continuation_steps']==5000 and not ck['smoke']
        assert ck['parent_sha256']==receipt['parent_sha256'] and ck['cfg']['name']=='peak-drop' and ck['cfg']['cache_precision']=='float16'
        assert all(ck['cfg'][k]==v for k,v in base['architecture'].items())
        assert all(torch.isfinite(v).all() for v in ck['state_dict'].values())
        filename='peak-drop-fold'+str(fold)+'.pt';shutil.copy2(path,PACKAGE/filename)
        models.append(dict(file=filename,fold=fold,arm='peak-drop',steps=45000,cache_precision='float16',checkpoint_sha256=sha(path)))
        receipts.append(receipt);del ck
    names=['model.py','features.py','fingerprint-cache.npy','fingerprint-cache-smiles.parquet','compatible_tail.py']
    for name in names:
        assert sha(parent/name)==base['files'][name],name
        shutil.copy2(parent/name,PACKAGE/name)
    source=(ROOT/'scripts/compatible_inference.py').read_text(encoding='utf-8')
    needle='expected_steps=200 if smoke else 40000';assert source.count(needle)==1
    source=source.replace(needle,"expected_steps=45000\n    assert not smoke and manifest['run_id']=='C18PK'")
    compile(source,'c18_pinned_inference','exec')
    (PACKAGE/'inference.py').write_text(source,encoding='utf-8',newline='\n')
    names+=['inference.py']+[m['file'] for m in models]
    manifest=dict(kind='compatible_scorer',run_id='C18PK',steps=45000,smoke=False,
        architecture=base['architecture'],models=models,files={n:sha(PACKAGE/n) for n in names},
        training_contract_sha256=sha(RUN/'source-contract.json'),evaluation_plan_sha256=sha(EVALUATION),
        packager_sha256=sha(__file__),parent_manifest_sha256=sha(parent/'c08-manifest.json'),
        matched_control_audit_sha256=sha(RUN/'matched-control-audit.json'),automatic_submission=False,
        selection='Final5000updates/fold only; no intermediate checkpoint or coefficient search',
        attribution='Compatible architecture/features from credited Saner/P01 kit; local C08 parents. Original source/model terms retained.')
    write(PACKAGE/'artifact-audit.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),
        models=2,all_finite=True,checkpoint_receipts=receipts,private_upload_required=True,
        inference_adaptation='Only cumulative-step/run-id acceptance differs from original compatible_inference; scoring arithmetic unchanged',
        quality_gate_passed=False,automatic_submission=False))
    write(existing,manifest)
    write(PACKAGE/'dataset-metadata.json',dict(id='clarkkitchen/casmi26-c18-peak-models-20261110',
        title='CASMI26 C18 NP balanced compatible models',licenses=[dict(name='CC-BY-NC-4.0')]))
    (PACKAGE/'README.md').write_text('Private C18PK models. Two final5000-update peak-drop continuations of pinned local C08 parents, based on the credited Saner/P01 compatible model and features. Parent/source/model hashes and original source terms are retained in the manifest. No model quality or leaderboard gain is asserted by packaging. No automatic upload or public submission.\n',encoding='utf-8')
    print('DONE C18 FINAL ARTIFACT AUDIT',json.dumps(dict(models=2,manifest_sha256=sha(existing))),flush=True)
    return manifest


def serving_module():
    manifest=package()
    spec=importlib.util.spec_from_file_location('_c18_pinned_serving',PACKAGE/'inference.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return manifest,module


if __name__=='__main__':package()
