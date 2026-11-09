"""Complete C09 packaging/development freeze once; no submission or refit."""
import csv,json,os,subprocess,time
from pathlib import Path
from datetime import datetime,timezone,timedelta

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/local/c09-20261109'
CLI=ROOT/'.venv/Scripts/python.exe';GPU=ROOT/'.venv-gpu/Scripts/python.exe'
def stamp():return datetime.now(timezone.utc).isoformat()
def ledger(run_id,result):
    path=ROOT/'runs.csv'
    with path.open(newline='',encoding='utf-8') as stream:reader=csv.DictReader(stream);fields=reader.fieldnames;rows=list(reader)
    item=next(r for r in rows if r['id']==run_id);item.update(status='complete',ended_utc=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M'),result=result)
    temporary=path.with_suffix('.c09.tmp')
    with temporary.open('w',newline='',encoding='utf-8') as stream:writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    temporary.replace(path)
def command(label,args):
    with (OUT/(label+'.log')).open('w') as stdout,(OUT/(label+'.err')).open('w') as stderr:
        subprocess.run(args,cwd=ROOT,stdout=stdout,stderr=stderr,check=True,timeout=1800)


def main():
    import msvcrt
    lock=(OUT/'training-finisher.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    deadline=datetime.now(timezone.utc)+timedelta(hours=6)
    while not (OUT/'results.json').exists():
        assert datetime.now(timezone.utc)<deadline and not (ROOT/'.local/compute-paused.json').exists()
        watchdog=OUT/'watchdog/status.json'
        if watchdog.exists():assert json.loads(watchdog.read_text()).get('action')!='needs_attention','Training supervisor needs attention'
        time.sleep(10)
    # Give the existing supervised trainer time to exit before inference owns
    # the only local GPU workload. Check actual process liveness explicitly.
    from training_monitor import process_alive
    status=json.loads((OUT/'status.json').read_text());pid=status.get('pid')
    while process_alive(pid):time.sleep(1)
    results=json.loads((OUT/'results.json').read_text())['models']
    ledger('C09T',f"2 folds x5000 continuation updates=10000 total (250 in pilot),45000 cumulative per fit;{sum(m['amp_skips'] for m in results)} AMP skips;peak reserved{max(m['gpu_peak_reserved_gib'] for m in results):.4f}GiB; actual-pipeline qualification pending")
    # The training supervisor was the sole account/C09V poller while training.
    # The read-only tracker takes ownership now, avoiding duplicate polling.
    tracker_path=OUT/'campaign-tracker.json'
    tracked=json.loads(tracker_path.read_text()) if tracker_path.exists() else {}
    if not process_alive(tracked.get('runner_pid')):
        with (OUT/'tracker-after-training.log').open('w') as stdout,(OUT/'tracker-after-training.err').open('w') as stderr:
            subprocess.Popen([str(CLI),'-u','scripts/track_c09_campaign.py'],cwd=ROOT,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
    command('package',[str(GPU),'-u','scripts/package_c09_models.py'])
    with (ROOT/'runs.csv').open(newline='',encoding='utf-8') as stream:ids={r['id'] for r in csv.DictReader(stream)}
    if 'C09D' not in ids:
        subprocess.run([str(CLI),'scripts/add_run.py','C09D','local-2070','diag','outputs/local/c09-20261109/development.log',
            'Freeze C09 weight only on previously used acquisition-clean development IDs; new80 confirmation never used; zero-weight tie keeps champion','1'],cwd=ROOT,check=True)
    if not (OUT/'development/selection.json').exists():command('development',[str(GPU),'-u','scripts/select_c09_candidate.py'])
    selection=json.loads((OUT/'development/selection.json').read_text());winner=selection['winner']
    ledger('C09D',f"{selection['development_queries']} previously used acquisition-clean development queries; selected weight{winner['weight']}; MRR{winner['paired']['baseline_mrr25']:.6f} to{winner['mrr25']:.6f}; qualifies for fresh confirmation={selection['qualify_for_confirmation']}; no promotion/submission")
    subprocess.run([str(CLI),'scripts/update_runs.py'],cwd=ROOT,check=True,timeout=300)
    print('DONE C09 TRAINING FINISHER',stamp(),json.dumps(dict(winner=winner,qualify_for_confirmation=selection['qualify_for_confirmation'])),flush=True)


if __name__=='__main__':main()
