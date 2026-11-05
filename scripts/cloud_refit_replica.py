"""Use two T4s for a production refit and an independently excluded v4 replica."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch


def main():
    work, inputs = Path('/kaggle/working'),Path('/kaggle/input')
    assert torch.cuda.device_count() == 2
    print('ACTUAL GPUS',[torch.cuda.get_device_name(i) for i in range(2)],flush=True)
    kit = next(p.parent for p in inputs.rglob('train.py') if p.parent.name == 'casmi_spectrum_model_v1')
    cache = next(p.parent for p in inputs.rglob('peaks.bin') if p.parent.name == 'cache')
    mol = pd.read_parquet(cache/'molecules.parquet')
    np_keys = np.load(next(inputs.rglob('np_unseen_candidate_keys.npz')),allow_pickle=False).files
    excluded = sorted(set(mol.loc[mol.split.eq('validation'),'key'].astype(str))|set(np_keys))
    plans = [dict(name='general_full20',gpu=0,stage='refit',seed=20,epochs=128,split='kit',width=256,heads=8,layers=4,embedding=192),
             dict(name='v4_replica_holdout0',gpu=1,stage='holdout',seed=0,epochs=109,split='np',width=256,heads=8,layers=3,embedding=256)]
    procs,handles = [],[]
    for plan in plans:
        out = work/plan['name']; out.mkdir(exist_ok=True)
        (out/'np_holdout_keys.txt').write_text('\n'.join(excluded)+'\n')
        training = np.arange(len(mol)) if plan['stage']=='refit' else np.flatnonzero(~mol.key.isin(excluded).to_numpy())
        manifest = dict(plan,training_identities=len(training),
                        heldout_identity_sha256=hashlib.sha256(('' if plan['stage']=='refit' else '\n'.join(excluded)).encode()).hexdigest(),
                        training_indices_sha256=hashlib.sha256(training.astype('<i8').tobytes()).hexdigest(),
                        checkpoint_policy='fixed final epoch; no selection on evaluation targets',
                        evaluate_on_holdout=plan['stage']=='holdout',
                        gpu_name=torch.cuda.get_device_name(plan['gpu']))
        (out/'training_manifest.json').write_text(json.dumps(manifest,indent=2))
        args=[sys.executable,'-u',str(Path(__file__).with_name('trial_worker.py')),json.dumps(plan)]
        handle=(work/(plan['name']+'.log')).open('w'); handles.append(handle)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(plan['gpu']),CASMI_KIT=str(kit),CASMI_CACHE=str(cache),CASMI_OUT=str(out),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
        procs.append(subprocess.Popen(args,env=env,stdout=handle,stderr=subprocess.STDOUT))
        print('LAUNCHED',json.dumps(manifest),'PID',procs[-1].pid,flush=True)
    while any(p.poll() is None for p in procs):
        time.sleep(60)
        for plan,p in zip(plans,procs):
            lines=(work/(plan['name']+'.log')).read_text().splitlines()
            print('PROGRESS',plan['name'],'rc',p.poll(),lines[-1][:450] if lines else '',flush=True)
    audits={}
    for plan,p in zip(plans,procs):
        assert p.returncode==0,(plan,p.returncode)
        out=work/plan['name']
        tag=plan['stage']+('_np' if plan['split']=='np' else '')
        report=json.loads((out/(tag+'_report.json')).read_text())
        assert len(report)==1
        name,result=next(iter(report.items()))
        assert result['completed_epochs']==plan['epochs'] and not result['time_budget_exhausted']
        assert len(result['history'])==plan['epochs']
        assert all(h['steps']==500 and not h['partial_epoch'] for h in result['history'])
        path=out/tag/name
        checkpoint=torch.load(path,map_location='cpu',weights_only=True)
        assert checkpoint['epoch']==plan['epochs']
        assert checkpoint['stage']==('full_refit' if plan['stage']=='refit' else 'holdout')
        assert checkpoint['cfg_overrides']=={k:plan[k] for k in ['width','heads','layers','embedding']}
        audits[plan['name']]=dict(completed_epochs=plan['epochs'],steps=plan['epochs']*500,
                                 seconds=result['seconds'],stage=checkpoint['stage'],
                                 checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (work/'completion_audit.json').write_text(json.dumps(audits,indent=2))
    print('DONE BOTH T4 JOBS',json.dumps(audits),flush=True)


if __name__=='__main__':
    main()
