"""Hourly read-only C08 progress runner; recovery is the separate supervisor.

Never trains or submits. The active task owns validation and the one authorized
submission after completion. Small JSON reads only; no model/data imports.
"""
import argparse,json,os,time
from datetime import datetime,timezone,timedelta
from pathlib import Path
from training_monitor import build_snapshot,process_alive,system_memory,gpu_sample

ROOT=Path(__file__).resolve().parents[1]


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2),encoding='utf-8');temp.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,default=ROOT/'configs/compatible-campaign-20261108.json')
    parser.add_argument('--once',action='store_true');args=parser.parse_args()
    campaign=json.loads(args.campaign.read_text());run=ROOT/campaign['training_output'];folder=run/'hourly'
    folder.mkdir(parents=True,exist_ok=True)
    import msvcrt
    lock=(folder/'runner.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    interval=campaign['tracking_interval_seconds'];assert interval>=3600
    while True:
        now=datetime.now(timezone.utc);snapshot=build_snapshot(run)
        row=dict(checked_utc=now.isoformat(),runner_pid=os.getpid(),run_id=campaign['run_id'],
            training_state=snapshot['state'],trainer_alive=snapshot['trainer_alive'],progress=snapshot['progress'],
            active=snapshot['active'],watchdog=snapshot['watchdog'],resources=snapshot['resources'],
            memory=system_memory(),gpu=gpu_sample(),next_report_utc=(now+timedelta(seconds=interval)).isoformat(),
            submission_status='Not submitted; training/validation/real Kaggle smoke required',
            approved_scoring_submissions=campaign['authorization']['maximum_scoring_submissions'])
        write(folder/'status.json',row)
        with (folder/'history.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
        if args.once or snapshot['state'] in ('validation_pending','complete','paused','budget_or_resource_stop') or snapshot['watchdog'].get('action') in ('needs_attention','budget_reached','paused'):return
        # Wake promptly on completion; do not log unchanged progress every minute.
        until=time.monotonic()+interval
        while time.monotonic()<until:
            if (ROOT/'.local/compute-paused.json').exists():break
            try:
                state=json.loads((run/'status.json').read_text()).get('state')
                watchdog=json.loads((run/'watchdog/status.json').read_text())
                if state in ('validation_pending','complete','budget_or_resource_stop') or watchdog.get('action') in ('needs_attention','budget_reached','paused'):break
            except (OSError,ValueError):pass
            time.sleep(min(30,max(0,until-time.monotonic())))
        if not process_alive(snapshot['watchdog'].get('pid')):
            # Observe and report a dead supervisor; never create a duplicate.
            current=build_snapshot(run)
            if current['state'] not in ('validation_pending','complete'):
                write(folder/'attention.json',dict(utc=datetime.now(timezone.utc).isoformat(),reason='Training supervisor is offline; inspect before restarting'))
                print('SUPERVISOR OFFLINE: inspect before restarting',flush=True);return


if __name__=='__main__':main()
