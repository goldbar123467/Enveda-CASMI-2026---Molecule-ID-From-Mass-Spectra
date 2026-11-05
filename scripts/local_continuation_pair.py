"""Keep the local GPU queue sequential: finish PL09, then replicate P09's two conditions."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]


def complete_report(path,epochs):
    try:
        report=json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):
        return False
    assert len(report)==1
    result=next(iter(report.values()))
    assert result['completed_epochs']==epochs and len(result['history'])==epochs
    assert not result['time_budget_exhausted']
    assert all(h['steps']==500 and not h['partial_epoch'] for h in result['history'])
    return True


def main():
    base=ROOT/'outputs/local/paired-native'
    out=base/'continuation20';out.mkdir(parents=True,exist_ok=True)
    status=out/'status.json'
    def record(**state):
        state['updated_utc']=datetime.now(timezone.utc).isoformat()
        status.write_text(json.dumps(state,indent=2))
        print(json.dumps(state),flush=True)
    record(state='waiting_for_native20_full128')
    deadline=time.monotonic()+3*3600
    while not complete_report(base/'native20/holdout_np_report.json',128):
        assert time.monotonic()<deadline,'Local native trial did not finish within expected training deadline'
        errors=base/'native20/native_w256_e128.err'
        assert not errors.exists() or 'Traceback' not in errors.read_text(errors='replace')
        time.sleep(5)
    parent=base/'holdout_np/casmi26_holdout_20_e128_heads8-layers4-width256.pt'
    expected=json.loads((ROOT/'outputs/local/professional/general20_terms.json').read_text())['checkpoint_sha256']
    assert hashlib.sha256(parent.read_bytes()).hexdigest()==expected
    python=ROOT/'.venv-gpu/Scripts/python.exe'
    cli=ROOT/'.venv/Scripts/python.exe'
    states={}
    for mode,rid in [('general','PL14'),('np','PL15')]:
        folder=out/mode;folder.mkdir(parents=True,exist_ok=True)
        (folder/'np_holdout_keys.txt').write_bytes((base/'np_holdout_keys.txt').read_bytes())
        log=out/(mode+'.log');err=out/(mode+'.err')
        ref=str(log.relative_to(ROOT)).replace('\\','/')
        purpose=f'Replicate fixed32x500 continuation of excluded general20 parent; mode={mode},LR2e-5,seed23,original loss; no epoch or weight sweep'
        subprocess.run([str(cli),str(ROOT/'scripts/add_run.py'),rid,'local-2070','train',ref,purpose,'2'],cwd=ROOT,check=True)
        env=dict(os.environ,CASMI_OUT=str(folder),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
        args=[str(python),'-u',str(ROOT/'scripts/finetune_sampling.py'),'--mode',mode,'--parent',str(parent),
              '--np-keys',str(base/'curated_np_keys.txt'),'--seed','23']
        with log.open('w') as stdout,err.open('w') as stderr:
            process=subprocess.Popen(args,cwd=ROOT,env=env,stdout=stdout,stderr=stderr)
            record(state='training',condition=mode,pid=process.pid,finished=states)
            code=process.wait()
        assert code==0,(mode,code,err.read_text(errors='replace')[-2000:])
        assert complete_report(folder/'holdout_np_report.json',32)
        manifest=json.loads((folder/'finetune_manifest.json').read_text())
        assert manifest['parent_sha256']==expected and not manifest['smoke']
        states[mode]=dict(state='complete',epochs=32,steps=16000,manifest=manifest)
    assert states['general']['manifest']['training_indices_sha256']==states['np']['manifest']['training_indices_sha256']
    record(state='complete',finished=states)
    print('DONE both local continuation conditions',flush=True)


if __name__=='__main__':
    main()
