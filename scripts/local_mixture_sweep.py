"""Two fixed mixture-dose trials on the free local GPU; no scoring submissions."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

from local_continuation_pair import complete_report

ROOT=Path(__file__).resolve().parents[1]


def main():
    base=ROOT/'outputs/local/paired-native';out=base/'mixture_sweep20';out.mkdir(exist_ok=True)
    parent=base/'holdout_np/casmi26_holdout_20_e128_heads8-layers4-width256.pt'
    expected=json.loads((ROOT/'outputs/local/professional/general20_terms.json').read_text())['checkpoint_sha256']
    assert hashlib.sha256(parent.read_bytes()).hexdigest()==expected
    assert json.loads((base/'continuation20/status.json').read_text())['state']=='complete'
    states={}
    for fraction,name,rid in [(.25,'q25','PL22'),(.75,'q75','PL23')]:
        folder=out/name;folder.mkdir(exist_ok=True)
        (folder/'np_holdout_keys.txt').write_bytes((base/'np_holdout_keys.txt').read_bytes())
        log=out/(name+'.log');err=out/(name+'.err')
        ref=str(log.relative_to(ROOT)).replace('\\','/')
        purpose=f'Prespecified NP mixture-dose sweep: fraction={fraction},same general20 parent/RNGseed23/LR2e-5/32x500; compare q.5. Requires other-parent confirmation before promotion.'
        subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),str(ROOT/'scripts/add_run.py'),rid,'local-2070','train',ref,purpose,'2'],cwd=ROOT,check=True)
        env=dict(os.environ,CASMI_OUT=str(folder),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
        args=[str(ROOT/'.venv-gpu/Scripts/python.exe'),'-u',str(ROOT/'scripts/finetune_sampling.py'),'--mode','np','--parent',str(parent),
              '--np-keys',str(base/'curated_np_keys.txt'),'--seed','23','--mixture',str(fraction)]
        with log.open('w') as stdout,err.open('w') as stderr:
            process=subprocess.Popen(args,cwd=ROOT,env=env,stdout=stdout,stderr=stderr)
            state=dict(state='training',condition=name,fraction=fraction,pid=process.pid,finished=states,updated_utc=datetime.now(timezone.utc).isoformat())
            (out/'status.json').write_text(json.dumps(state,indent=2));print(json.dumps(state),flush=True)
            code=process.wait()
        assert code==0,(name,code,err.read_text(errors='replace')[-2000:])
        assert complete_report(folder/'holdout_np_report.json',32)
        manifest=json.loads((folder/'finetune_manifest.json').read_text())
        assert manifest['parent_sha256']==expected and manifest['mixture_fraction']==fraction and not manifest['smoke']
        states[name]=dict(epochs=32,steps=16000,manifest=manifest)
    (out/'status.json').write_text(json.dumps(dict(state='complete',finished=states,updated_utc=datetime.now(timezone.utc).isoformat()),indent=2))
    print('DONE both prespecified mixture doses',flush=True)


if __name__=='__main__':
    main()
