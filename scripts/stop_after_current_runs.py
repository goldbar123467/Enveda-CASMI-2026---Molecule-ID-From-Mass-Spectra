"""Honor the user's stop boundary; observe existing jobs and never launch work."""
import argparse
import json
import os
import time
from datetime import datetime,timezone
from pathlib import Path
from update_runs import process_running

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/local/stop-after-runs-20261110'
REQUEST=ROOT/'.local/stop-after-current-runs.json'
COLLECTORS={
    'clarkkitchen/casmi26-c29-fixed-public-replay-development':ROOT/'outputs/local/c29-public-replay-20261110/collector-status.json',
    'clarkkitchen/casmi26-c31-reserved-public-replay':ROOT/'outputs/local/c31-reserved-public-20261110/collector-status.json'}


def stamp():return datetime.now(timezone.utc).isoformat()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2),encoding='utf-8');tmp.replace(path)


def main():
    import msvcrt
    from kaggle.api.kaggle_api_extended import KaggleApi
    from finish_compatible_campaign import status_name
    p=argparse.ArgumentParser();p.add_argument('--until-utc',required=True);args=p.parse_args()
    deadline=datetime.fromisoformat(args.until_utc.replace('Z','+00:00'))
    OUT.mkdir(exist_ok=True);lock=(OUT/'controller.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    api=KaggleApi();api.authenticate();terminal={'COMPLETE','ERROR','CANCELLED','CANCEL_ACKNOWLEDGED','CANCELED'}
    while datetime.now(timezone.utc)<deadline:
        request=json.loads(REQUEST.read_text())
        if request.get('cancelled_by_user_resume'):return
        assert request['no_new_training'] and request['no_new_cloud_runs'] and request['no_new_submissions']
        states={};errors={};collectors={}
        for ref,path in COLLECTORS.items():
            try:states[ref]=status_name(api.kernels_status(ref))
            except Exception as exc:errors[ref]=type(exc).__name__
            if path.exists():
                s=json.loads(path.read_text());collectors[ref]=dict(pid=s['pid'],live=process_running(s['pid']),state=s.get('state'))
        condition=not errors and len(states)==len(COLLECTORS) and all(s in terminal for s in states.values())
        status=dict(checked_utc=stamp(),pid=os.getpid(),deadline_utc=deadline.isoformat(),state='draining',
            cloud=states,observation_errors=errors,collectors=collectors,new_work_permitted=False,
            reserved_metrics_read=False,automatic_submission=False)
        if request['scope']!='both_current_kaggle_runs':
            status.update(state='needs_scope_action',reason='Stop scope changed; no automatic cancellation of another live cloud job')
            write(OUT/'status.json',status);return
        if condition and len(collectors)==len(COLLECTORS) and all(not c['live'] for c in collectors.values()):
            write(ROOT/'.local/compute-paused.json',dict(paused_utc=stamp(),phase='stopped',reason='User requested stop after current full runs',
                cloud_terminal_states=states,resume_requires_explicit_user_instruction=True))
            write(ROOT/'.local/c09-tracker-stop.json',dict(requested_utc=stamp(),reason='Current full runs finished; user stop boundary reached'))
            status.update(state='stopped',completed_utc=stamp(),goal_pause_ready=True,
                next_action='Pause the Codex goal per user request; do not pursue silver or launch/evaluate/submit anything further.')
            write(OUT/'status.json',status);print('DONE USER-REQUESTED STOP BOUNDARY',json.dumps(states),flush=True);return
        write(OUT/'status.json',status)
        print('STOP BOUNDARY WAIT',stamp(),json.dumps(dict(cloud=states,collectors=collectors,errors=errors)),flush=True)
        time.sleep(60)
    write(OUT/'attention.json',dict(checked_utc=stamp(),reason='Observation deadline; no job termination inferred',new_work_permitted=False))
    raise TimeoutError('Recheck same jobs; never requeue or launch new work under this stop request')


if __name__=='__main__':main()
