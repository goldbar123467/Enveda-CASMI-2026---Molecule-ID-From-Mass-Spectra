"""Two-T4 general/NP-mixture continuation of the same excluded general21 parent."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pandas as pd
import torch

CURATED_SHA256='19d59131c440e04b98f43b7307cf53a65d41c2e90680937668170abd3723e420'


def main(production=False):
    inputs,work=Path('/kaggle/input'),Path('/kaggle/working')
    assert torch.cuda.device_count()==2
    print('ACTUAL GPUS',[torch.cuda.get_device_name(i) for i in range(2)],flush=True)
    kit=next(p.parent for p in inputs.rglob('train.py') if p.parent.name=='casmi_spectrum_model_v1')
    cache=next(p.parent for p in inputs.rglob('peaks.bin') if p.parent.name=='cache')
    pattern='casmi26_seed_20_e128_embedding192-heads8-layers4-width256.pt' if production else 'casmi26_holdout_21_e128_heads8-layers4-width256.pt'
    parent_folder='general_full20' if production else 'general'
    parents=[p for p in inputs.rglob(pattern) if parent_folder in p.parts]
    assert len(parents)==1
    parent=parents[0]
    excluded=parent.parent.parent/'np_holdout_keys.txt'
    if not production:
        assert len(set(excluded.read_text().split()))==27600
    cols=['inchikey14','np_coconut','np_lotus','src_natural_product']
    pop=pd.read_csv(next(inputs.rglob('pool_popularity.csv')),usecols=cols)
    assert all(set(pop[c].unique())<={0,1} for c in cols[1:])
    keys=set(pop.loc[pop.np_coconut.astype(bool)|pop.np_lotus.astype(bool)|pop.src_natural_product.astype(bool),'inchikey14'].astype(str))
    keys|=set(pd.read_parquet(next(inputs.rglob('candidates_lotus.parquet')),columns=['inchikey14']).inchikey14.astype(str))
    text='\n'.join(sorted(keys))
    assert hashlib.sha256(text.encode()).hexdigest()==CURATED_SHA256
    key_file=work/'curated_np_keys.txt';key_file.write_text(text+'\n')
    procs={};handles=[]
    for gpu,mode in enumerate(['general','np']):
        out=work/('production' if production else 'continuation')/mode;out.mkdir(parents=True,exist_ok=True)
        if not production:
            (out/'np_holdout_keys.txt').write_bytes(excluded.read_bytes())
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CASMI_KIT=str(kit),CASMI_CACHE=str(cache),CASMI_OUT=str(out),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
        args=[sys.executable,'-u',str(Path(__file__).with_name('finetune_sampling.py')),'--mode',mode,'--parent',str(parent),'--np-keys',str(key_file),
              '--seed','23' if production else '22','--stage','refit' if production else 'holdout']
        handle=(work/(mode+'.log')).open('w');handles.append(handle)
        procs[mode]=subprocess.Popen(args,env=env,stdout=handle,stderr=subprocess.STDOUT)
        print('LAUNCHED',mode,'GPU',gpu,'parent',parent,'PID',procs[mode].pid,flush=True)
    while any(p.poll() is None for p in procs.values()):
        time.sleep(60)
        for mode,p in procs.items():
            lines=(work/(mode+'.log')).read_text().splitlines()
            print(mode,'rc',p.poll(),lines[-1][:450] if lines else '',flush=True)
    audits={}
    for mode,p in procs.items():
        assert p.returncode==0,(mode,p.returncode,(work/(mode+'.log')).read_text()[-2500:])
        out=work/('production' if production else 'continuation')/mode
        tag='refit' if production else 'holdout_np'
        report=json.loads((out/(tag+'_report.json')).read_text());assert len(report)==1
        name,result=next(iter(report.items()))
        assert result['completed_epochs']==32 and len(result['history'])==32 and not result['time_budget_exhausted']
        assert all(h['steps']==500 and not h['partial_epoch'] for h in result['history'])
        checkpoint=out/tag/name
        ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
        assert ck['stage']==('full_refit' if production else 'holdout') and ck['epoch']==32 and ck['initialization_parent_epoch']==128 and not ck['smoke']
        audits[mode]=dict(epochs=32,steps=16000,seconds=result['seconds'],checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),manifest=json.loads((out/'finetune_manifest.json').read_text()))
    assert audits['general']['manifest']['parent_sha256']==audits['np']['manifest']['parent_sha256']
    assert audits['general']['manifest']['training_indices_sha256']==audits['np']['manifest']['training_indices_sha256']
    (work/'completion_audit.json').write_text(json.dumps(audits,indent=2))
    print('DONE both fixed continuation conditions',json.dumps(audits),flush=True)


if __name__=='__main__':
    main()
