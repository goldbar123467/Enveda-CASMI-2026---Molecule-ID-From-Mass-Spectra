"""Bounded RTX2070 diagnostic of the actual empty-pool Engine2 recovery."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS']:
    os.environ[key]='2'
import argparse,hashlib,json,subprocess,sys,time
from datetime import datetime,timezone,timedelta
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/research/c20-engine-recovery-20261110'
sys.path.insert(0,str(OUT/'engine-site'))
sys.path.insert(0,str(OUT/'local-eng'))
sys.path.insert(0,str(ROOT/'scripts'))


def stamp():return datetime.now(timezone.utc).isoformat()


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        while block:=stream.read(4*1024*1024):h.update(block)
    return h.hexdigest()


def write(name,value):
    p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2));t.replace(p)


def worker():
    import numpy as np,pandas as pd,torch,pyarrow as pa
    from rdkit import rdBase
    from empty_pool_recovery import augment
    from training_monitor import system_memory
    import casmi_engine as E
    torch.set_num_threads(2);torch.set_num_interop_threads(1);pa.set_cpu_count(2)
    assert rdBase.rdkitVersion=='2025.03.6' and torch.cuda.is_available() and '2070' in torch.cuda.get_device_name()
    torch.cuda.set_per_process_memory_fraction(.7)
    assert system_memory()['free_gib']>=4
    cfg=json.loads((OUT/'local-engine-contract.json').read_text())
    for path,pin in cfg['files_sha256'].items():assert digest(ROOT/path)==pin,path
    mid='C20Q_3f38081e510120c5bad2'
    q=pd.read_parquet(ROOT/'outputs/publish/c20-negative-casmi2022-20261110/validation-queries.parquet')
    q=q[q.molecule_id.eq(mid)];assert len(q)==1
    baseline=pd.read_csv(ROOT/'outputs/forks/c20-negative-development-v1/baseline_submission.csv',dtype=str,keep_default_na=False).set_index('molecule_id')
    pc=json.loads((ROOT/'outputs/forks/c20-negative-development-v1/pc_lists.json').read_text())
    old=json.loads((ROOT/'outputs/forks/c20-negative-development-v1/eng_lists.json').read_text())
    assert mid not in old and not pc[mid]['pc']
    # Frozen cell43 takes BASE unchanged when this PC proposal list is empty;
    # its saved pre-guard row therefore supplies the real BASE top25 proposals.
    proposals={mid:baseline.loc[mid,'smiles'].split(';')};assert len(proposals[mid])==25 and 'CCO' not in proposals[mid]
    query=OUT/'local-recovery-query.parquet';sample=OUT/'local-recovery-sample.csv'
    q.to_parquet(query,index=False);pd.DataFrame(dict(molecule_id=[mid],smiles='CCO')).to_csv(sample,index=False)
    write('local-engine-progress.json',dict(updated_utc=stamp(),pid=os.getpid(),stage='loading original library and candidate pool',labels_read=False))
    original_build=E.build_pool
    def recovered_pool(library,workers):
        pool=original_build(library,workers)
        assert len(pool['mass'])==1137233,'Original v1 pool count mismatch; never call this a faithful recovery'
        write('local-engine-progress.json',dict(updated_utc=stamp(),pid=os.getpid(),stage='fingerprinting real generated proposals',original_pool=len(pool['mass']),labels_read=False))
        return augment(pool,q,proposals,E,OUT/'local-engine-pool-audit.json')
    E.build_pool=recovered_pool
    E.RANK.W_A=(.35,.55);E.RANK.SEEDS=(0,1);E.CFG.N_ANALOG=200
    models=sorted(str(p) for p in (OUT/'assets/casmi26-fp-models-v2').glob('fp_*.pt'));assert len(models)==2
    _,records,scores=E.main(str(query),str(ROOT/'outputs/competition-raw/train.parquet'),str(sample),
        E.find('sim_rank_rows_nofp.npz'),E.find('rank_train.npz'),models,workers=2,w_pv=.88)
    blend=E.blend_scores(scores['pv'],scores['ours'],.88);result={}
    for record in records:
        name=record['mid'];assert name in blend and len(record['cand'])>0
        indices=np.argsort(-blend[name],kind='mergesort')[:80];smiles=[];keys=[]
        for candidate in record['cand'][indices]:
            s=E.POOL['smiles'][candidate];key=E.canon_key(s)
            if not key or key in keys:continue
            smiles.append(str(s));keys.append(str(key))
            if len(smiles)>=40:break
        assert smiles and len(smiles)==len(keys) and 'CCO' not in smiles
        result[str(name)]=dict(smiles=smiles,keys=keys)
    assert set(result)=={mid}
    write('local-engine-recovered-lists.json',result)
    import sklearn,numba
    write('local-engine-audit.json',dict(completed_utc=stamp(),recovered_queries=1,real_generated_proposals=25,
        original_pool_structures=1137233,real_fingerprints_and_both_rankers_executed=True,
        GPU=torch.cuda.get_device_name(),torch=torch.__version__,rdkit=rdBase.rdkitVersion,
        sklearn=sklearn.__version__,numba=numba.__version__,pool_audit_sha256=digest(OUT/'local-engine-pool-audit.json'),
        recovered_lists_sha256=digest(OUT/'local-engine-recovered-lists.json'),contract_sha256=digest(OUT/'local-engine-contract.json'),
        limitation='Local diagnostic uses Torch2.11 and sklearn1.6.1; exact cloud output parity and complete80-pipeline receipt are still required. This is not a qualified full-P01 validation.',
        labels_read=False,quality_metrics_computed=False,C13_confirmation_metrics_read=False,submission_qualified=False))
    print('DONE C20 LOCAL ENGINE DIAGNOSTIC',flush=True)


def supervise():
    import msvcrt
    lock=(OUT/'local-engine.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    assert not (OUT/'local-engine-intent.json').exists(),'Inspect existing process/artifacts before any recovery'
    assert json.loads((OUT/'asset-status.json').read_text())['state']=='complete'
    from training_monitor import system_memory
    assert system_memory()['free_gib']>=8
    deadline=datetime.fromisoformat(json.loads((OUT/'local-engine-contract.json').read_text())['deadline_utc'])
    assert datetime.now(timezone.utc)<deadline
    write('local-engine-intent.json',dict(started_utc=stamp(),pid=os.getpid(),deadline_utc=deadline.isoformat(),
        source_sha256=digest(Path(__file__)),labels_read=False,quality_metrics_computed=False,automatic_submission=False))
    with (OUT/'local-engine.log').open('w',encoding='utf-8') as stdout,(OUT/'local-engine.err').open('w',encoding='utf-8') as stderr:
        child=subprocess.Popen([str(ROOT/'.venv-gpu/Scripts/python.exe'),str(Path(__file__)),'--child'],cwd=ROOT,stdout=stdout,stderr=stderr)
        try:
            while child.poll() is None:
                memory=system_memory()
                write('local-engine-status.json',dict(checked_utc=stamp(),pid=os.getpid(),child_pid=child.pid,state='running',
                    free_ram_gib=memory['free_gib'],deadline_utc=deadline.isoformat(),C13_confirmation_metrics_read=False,automatic_submission=False))
                assert memory['free_gib']>=4,'RAM below configured4GiB floor'
                assert datetime.now(timezone.utc)<deadline,'Local diagnostic wall budget ended'
                assert not (ROOT/'.local/compute-paused.json').exists()
                time.sleep(10)
            assert child.returncode==0,'Local Engine2 diagnostic failed; inspect logs before any recovery'
        except BaseException:
            # A Windows venv redirector owns the actual Python worker and its
            # pool. Terminating only the redirector can leave that work alive.
            # Kill this exact spawned process tree, never an arbitrary Python.
            if child.poll() is None:
                subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],check=True,capture_output=True,timeout=30)
                child.wait(timeout=30)
            raise
    assert (OUT/'local-engine-audit.json').exists()
    write('local-engine-status.json',dict(completed_utc=stamp(),pid=os.getpid(),child_pid=child.pid,state='complete',
        C13_confirmation_metrics_read=False,automatic_submission=False,submission_qualified=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--child',action='store_true');args=parser.parse_args()
    try:worker() if args.child else supervise()
    except Exception as error:
        write('local-engine-attention'+('-child' if args.child else '')+'.json',dict(checked_utc=stamp(),pid=os.getpid(),
            error=type(error).__name__+': '+str(error),automatic_retry=False,C13_confirmation_metrics_read=False))
        raise
