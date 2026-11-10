"""Finish only the two explicitly approved trials, sequentially; never train."""
import argparse,json,os,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
from submit_past_trial_exports import ROOT,OUT,LEDGER,COMP,write,stamp
from c09_submission_guard import normalize,TERMINAL

def result_ready(row):
    return row['status'] in TERMINAL and (row.get('success') or row.get('error') or row['status'] in ['ERROR','CANCELLED','CANCELED'])

def main():
    import msvcrt
    from kaggle.api.kaggle_api_extended import KaggleApi
    p=argparse.ArgumentParser();p.add_argument('--until-utc',required=True);a=p.parse_args()
    deadline=datetime.fromisoformat(a.until_utc.replace('Z','+00:00'))
    lock=(OUT/'monitor.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    assert LEDGER.exists() and len(json.loads(LEDGER.read_text())['submissions'])>=1,'First trial must be reviewed and submitted manually'
    api=KaggleApi();api.authenticate()
    while datetime.now(timezone.utc)<deadline:
        if (ROOT/'.local/compute-paused.json').exists():
            write(OUT/'monitor-status.json',dict(checked_utc=stamp(),pid=os.getpid(),state='stopped_user_pause',no_more_submissions=True));return
        try:raw=api.competition_submissions(COMP)
        except Exception as exc:
            write(OUT/'monitor-status.json',dict(checked_utc=stamp(),pid=os.getpid(),state='observing',error_type=type(exc).__name__,terminal_inferred=False));time.sleep(60);continue
        normalized=[normalize(r) for r in raw];ledger=json.loads(LEDGER.read_text());assert len(ledger['submissions'])<=2
        for row in ledger['submissions']:
            hits=[r for r in normalized if r['reference']==str(row['reference'])];assert len(hits)==1
            row.update(hits[0],actual_checked_utc=stamp())
        write(LEDGER,ledger)
        first=next(r for r in ledger['submissions'] if r['slot']=='P18')
        second=next((r for r in ledger['submissions'] if r['slot']=='P19'),None)
        if result_ready(first) and second is None:
            # The guarded submitter rechecks global pending, quota, immutable
            # qualification, exact remote version and both durable intents.
            result=subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'-u','scripts/submit_past_trial_exports.py','--slot','P19'],cwd=ROOT)
            assert result.returncode==0,'Second trial requires inspection; never blindly retry an intent'
            ledger=json.loads(LEDGER.read_text());second=next(r for r in ledger['submissions'] if r['slot']=='P19')
        status=dict(checked_utc=stamp(),pid=os.getpid(),deadline_utc=deadline.isoformat(),state='waiting_actual_scores',
            accepted=len(ledger['submissions']),maximum_submissions=2,maximum_pending=1,submissions=ledger['submissions'],
            training_performed=False,hourly_campaign_resumed=False)
        if second is not None and all(result_ready(r) for r in ledger['submissions']):
            status.update(state='complete',completed_utc=stamp(),no_further_work=True)
            write(OUT/'final-results.json',status);write(OUT/'monitor-status.json',status)
            write(ROOT/'.local/compute-paused.json',dict(paused_utc=stamp(),phase='two_trials_finished',reason='Two requested past-work scoring trials reached actual terminal results; no further work authorized',resume_requires_explicit_user_instruction=True))
            print('DONE TWO REQUESTED SCORING TRIALS',json.dumps(ledger['submissions']),flush=True);return
        write(OUT/'monitor-status.json',status);print('TWO TRIALS WAIT',json.dumps(status),flush=True);time.sleep(60)
    write(OUT/'monitor-attention.json',dict(checked_utc=stamp(),reason='Observation deadline; do not infer completion or re-submit',automatic_retry=False))

if __name__=='__main__':
    # A temporary thread-scoped system sleep inhibitor keeps the authorized
    # sequential handoff alive. The display can sleep; no power plan changes.
    import ctypes
    if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:main()
    except BaseException as exc:
        write(OUT/'monitor-attention.json',dict(checked_utc=stamp(),error_type=type(exc).__name__,reason=str(exc),automatic_retry=False));raise
    finally:
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
