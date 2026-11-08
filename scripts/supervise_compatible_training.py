"""Own one C08 child process; audit health every30 minutes, at most3 restarts.

Resource fallbacks are predeclared microbatches. No source edits, submissions or
new scientific recipes occur in recovery. Restart only from atomic checkpoints.
"""
import argparse,csv,hashlib,json,os,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone,timedelta

ROOT=Path(__file__).resolve().parents[1]


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2),encoding='utf-8');temp.replace(path)


def stamp():return datetime.now(timezone.utc).isoformat()


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        while chunk:=stream.read(4*1024*1024):h.update(chunk)
    return h.hexdigest()


def stop(child):
    if child is not None and child.poll() is None:
        subprocess.run(['taskkill.exe','/PID',str(child.pid),'/T','/F'],capture_output=True,timeout=30)
        child.wait(timeout=30)


def ledger(spec,state,result):
    path=ROOT/'runs.csv'
    with path.open(newline='',encoding='utf-8') as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
    found=[r for r in rows if r['id']==spec['run_id']]
    if not found:
        subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),str(ROOT/'scripts/add_run.py'),spec['run_id'],'local-2070','train',
            spec['output']+'/train-0.log','C08 controlled320 scorer: precision x hard-negative recipe,2 common identity/acquisition folds,24h cap; no automatic submission','1'],cwd=ROOT,check=True)
        return
    found[0].update(status=state,result=result)
    if state in ('complete','error','stopped'):found[0]['ended_utc']=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')
    temp=path.with_suffix('.csv.tmp')
    with temp.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    temp.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,default=ROOT/'configs/compatible-training-20261108.json')
    parser.add_argument('--check',action='store_true',help='Validate launch readiness without starting a trainer or wall clock');args=parser.parse_args()
    spec=json.loads(args.plan.read_text());out=ROOT/spec['output'];out.mkdir(parents=True,exist_ok=True)
    import msvcrt
    lock=(out/'supervisor.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    assert (out/'prepared.json').exists(),'Run --prepare-only and the isolated smoke first'
    smoke=ROOT/'outputs/local/smoke-compatible-20261108/results.json'
    assert smoke.exists(),'200-step smoke for every arm/fold required'
    smoke_models=json.loads(smoke.read_text())['models']
    assert {(m['arm'],m['fold']) for m in smoke_models}=={(a['name'],f) for a in spec['arms'] for f in spec['folds']}
    assert len(smoke_models)==len(spec['arms'])*len(spec['folds']) and all(m['steps']>=200 and m['smoke'] and m['amp_skips']==0 and m['heldout_in_training']==0 for m in smoke_models),'Every smoke model must pass'
    assert not (ROOT/'.local/compute-paused.json').exists(),'Project paused'
    from long_run_watchdog import inventory
    assert not [p for p in inventory() if str(ROOT/'scripts/compatible_long_run.py').lower() in (p.get('CommandLine') or '').lower()], 'Inspect existing trainer before launch'
    sources=['scripts/compatible_long_run.py','scripts/compatible_training.py','scripts/precision_cache.py',
             'scripts/supervise_compatible_training.py','outputs/local/kit/features.py','outputs/local/kit/model.py']
    pins={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}
    prepared=json.loads((out/'prepared.json').read_text())
    assert all(pins[p]==v for p,v in prepared['source_sha256'].items()),'Preparation used different scientific sources'
    smoke_manifest=json.loads(smoke.with_name('manifest.json').read_text())
    assert all(pins[p]==v for p,v in smoke_manifest['source_sha256'].items()),'Smoke used different sources'
    production=dict(spec);pilot=dict(smoke_manifest['spec'])
    for key in ['run_id','output']:production.pop(key);pilot.pop(key)
    for key in ['steps','warmup_steps','checkpoint_interval']:production['training']=dict(production['training']);pilot['training']=dict(pilot['training']);production['training'].pop(key,None);pilot['training'].pop(key,None)
    assert production==pilot,'Smoke and production recipe differ beyond run length'
    cache=ROOT/spec['cache'];cached=json.loads((cache/'prepared.json').read_text());data_pins={}
    for name,item in cached['arrays'].items():
        path=cache/(name+'.npy');value=digest(path);assert value==item['sha256'],'Cache array bytes changed: '+name
        data_pins[str(path.relative_to(ROOT))]=value
    assert digest(cache/'valid.npy')==cached['valid_sha256'],'Valid-row mask changed'
    for fold in spec['folds']:
        path=cache/f'pools-fold-{fold}.npy';receipt=json.loads(path.with_suffix('.json').read_text());value=digest(path)
        assert value==receipt['sha256'],'Negative pool bytes changed';data_pins[str(path.relative_to(ROOT))]=value
    for path in [cache/'valid.npy',cache/'selected.parquet',ROOT/spec['split_groups'],ROOT/spec['base_cache']/'fingerprints.npy',ROOT/spec['base_cache']/'molecules.parquet']:
        data_pins[str(path.relative_to(ROOT))]=digest(path)
    ready=out/'launch-ready.json';data_contract=dict(data_sha256=data_pins,scientific_source_sha256=prepared['source_sha256'])
    if ready.exists():assert json.loads(ready.read_text())==data_contract,'Prepared/piloted input bytes changed'
    else:write(ready,data_contract)
    if args.check:
        print(json.dumps(dict(ready=True,models=len(spec['arms'])*len(spec['folds']),maximum_hours=spec['maximum_hours'],
            micro_batch_options=spec['micro_batch_options'],scientific_sources_match=True,smoke_models=len(json.loads(smoke.read_text())['models']),
            input_byte_pins_verified=len(data_pins),full_training_started=(out/'launch.json').exists(),automatic_submission=False)),flush=True);return
    launch=out/'launch.json'
    if not launch.exists():write(launch,dict(started_utc=stamp()))
    deadline=datetime.fromisoformat(json.loads(launch.read_text())['started_utc'])+timedelta(hours=spec['maximum_hours'])
    contract=dict(source_sha256=pins,data_sha256=data_pins,plan=spec,deadline_utc=deadline.isoformat())
    pin_path=out/'supervisor-pins.json'
    if pin_path.exists():assert json.loads(pin_path.read_text())==contract,'Supervisor source/plan changed'
    else:write(pin_path,contract)
    ledger(spec,'running','Prepared8-model factorial; awaiting training steps')
    watchdog=out/'watchdog/status.json';old=json.loads(watchdog.read_text()) if watchdog.exists() else {}
    restarts=old.get('restarts',0);micro_index=old.get('micro_index',0);previous_error=old.get('last_error');child=None
    action='needs_attention';reason='Supervisor stopped unexpectedly'
    def record(action,**extra):
        row=dict(action=action,pid=os.getpid(),trainer_pids=[child.pid] if child and child.poll() is None else [],
                 checked_utc=stamp(),next_check_utc=(datetime.now(timezone.utc)+timedelta(minutes=30)).isoformat(),
                 deadline_utc=deadline.isoformat(),restarts=restarts,micro_index=micro_index,micro_batch=spec['micro_batch_options'][micro_index],last_error=previous_error,**extra)
        write(watchdog,row);print(json.dumps(row),flush=True)
    try:
        while True:
            if (ROOT/'.local/compute-paused.json').exists():action='paused';reason='Project pause requested';break
            if datetime.now(timezone.utc)>=deadline:action='budget_reached';reason='Original24h wall deadline';break
            assert json.loads(args.plan.read_text())==spec and all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==v for p,v in pins.items()),'Source or plan changed'
            if child is None or child.poll() is not None:
                if child is not None:
                    status=json.loads((out/'status.json').read_text()) if (out/'status.json').exists() else {}
                    if child.returncode==0 and status.get('state')=='validation_pending':
                        models=json.loads((out/'results.json').read_text())['models'];ledger(spec,'complete',f"{len(models)} models;{sum(m['steps'] for m in models)} successful updates; actual-pipeline validation pending; no submission")
                        action='complete';reason='Training complete; validation pending';break
                    failure=json.loads((out/'failure.json').read_text()) if (out/'failure.json').exists() else {}
                    error=(out/f'train-{restarts}.err').read_text(errors='replace');signature=hashlib.sha256(error[-6000:].encode()).hexdigest()
                    is_oom=child.returncode!=75 and failure.get('oom',False)
                    if is_oom and micro_index+1<len(spec['micro_batch_options']):micro_index+=1
                    elif signature==previous_error and child.returncode!=75:reason='Repeated failure requires inspection';break
                    if restarts>=3:reason='Three-restart limit reached';break
                    if child.returncode==75:
                        record('resource_wait',reason='Allow up to30min for RAM to recover before checkpoint restart')
                        from training_monitor import system_memory
                        for _ in range(360):
                            if (ROOT/'.local/compute-paused.json').exists() or datetime.now(timezone.utc)>=deadline:break
                            if system_memory().get('free_gib',100)>=spec['minimum_free_ram_gib']:break
                            time.sleep(5)
                        if (ROOT/'.local/compute-paused.json').exists() or datetime.now(timezone.utc)>=deadline:continue
                    restarts+=1;previous_error=signature
                    record('recovering',reason='Fresh process from checkpoint; smaller microbatch' if is_oom else 'Fresh process from checkpoint')
                stdout=(out/f'train-{restarts}.log').open('w',encoding='utf-8');stderr=(out/f'train-{restarts}.err').open('w',encoding='utf-8')
                env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
                child=subprocess.Popen([str(ROOT/'.venv-gpu/Scripts/python.exe'),'-u',str(ROOT/'scripts/compatible_long_run.py'),
                    '--plan',str(args.plan),'--micro-batch',str(spec['micro_batch_options'][micro_index])],cwd=ROOT,env=env,stdout=stdout,stderr=stderr,
                    creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS)
                stdout.close();stderr.close()
            record('healthy')
            # Check completion promptly. Audit/repair progress only every30min.
            for _ in range(1800):
                if child.poll() is not None or (ROOT/'.local/compute-paused.json').exists() or datetime.now(timezone.utc)>=deadline:break
                time.sleep(1)
            if child.poll() is None:
                status=json.loads((out/'status.json').read_text()) if (out/'status.json').exists() else {}
                updated=status.get('updated_utc');stale=not updated or (datetime.now(timezone.utc)-datetime.fromisoformat(updated)).total_seconds()>900
                if stale:record('restart_stalled',reason='No heartbeat for15minutes');stop(child)
    finally:
        stop(child);record(action,reason=reason)
        if action!='complete':ledger(spec,'error' if action=='needs_attention' else 'stopped',reason)
        subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),str(ROOT/'scripts/update_runs.py')],cwd=ROOT,timeout=120,check=False)


if __name__=='__main__':main()
