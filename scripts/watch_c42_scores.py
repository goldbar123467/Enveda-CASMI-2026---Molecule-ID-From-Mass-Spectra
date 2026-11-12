"""Observe accepted CPU scoring, release its frozen paired GPU trial, then stop.

Start only after qualification and actual CPU acceptance. All new submissions
go through the existing source-bound two-entry guard; this observer cannot fit,
choose another recipe, consume C13 or create a third entry.
"""
import argparse,csv,hashlib,json,os,time
from pathlib import Path
from datetime import datetime,timedelta,timezone
from c09_submission_guard import normalize,TERMINAL

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/local/c42-two-new-submissions-20261111'
LEDGER=ROOT/'.local/c42-two-new-submissions-20261111.json'
COMP='enveda-CASMI26-molecule-id-mass-spectra'


def stamp():return datetime.now(timezone.utc).isoformat()


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2),encoding='utf8');tmp.replace(path)


def register_actual(record):
    path=ROOT/'runs.csv'
    with path.open(newline='',encoding='utf8') as stream:reader=csv.DictReader(stream);fields=reader.fieldnames;rows=list(reader)
    found=False
    for row in rows:
        if row['platform']=='kaggle-submission' and row['ref']==str(record['reference']):
            found=True;row['status']=record['status'].lower();row['finished_utc']=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')
            row['result']='Actual '+record['status']+' public score '+str(record['public_score'])+'; error '+str(record['error'])+'. Fixed approved C42trial; no further entry beyond the pair.'
    assert found,'Accepted scoring entry is not registered in runs.csv'
    temporary=path.with_suffix('.csv.tmp')
    with temporary.open('w',newline='',encoding='utf8') as stream:writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    temporary.replace(path)


def main(release_second):
    assert release_second,'Explicit --release-second is required for the already-approved second trial'
    import msvcrt
    with (OUT/'score-observer.lock').open('a+b') as lock:
        lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        return observe_accepted_pair()


def observe_accepted_pair():
    from kaggle.api.kaggle_api_extended import KaggleApi
    from submit_c42_pair import submit
    assert not (OUT/'actual-final-results.json').exists()
    qpath=OUT/'submission-qualification.json';q=json.loads(qpath.read_text());qsha=sha(qpath)
    assert q['decision']=='execution_qualified_fixed_exploratory_pair' and q['order']==['cpu','gpu'] and q['maximum_new_accepted_submissions']==2
    assert q['artifact_sha256'][str(Path(__file__).relative_to(ROOT))]==sha(__file__),'Observer source differs from qualification'
    ledger=json.loads(LEDGER.read_text());assert len([r for r in ledger['submissions'] if r['trial']=='cpu'])==1,'First CPUentry must already be accepted'
    assert len(ledger['submissions'])<=2 and all(r['qualification_sha256']==qsha for r in ledger['submissions'])
    deadline=datetime.now(timezone.utc)+timedelta(hours=24);api=KaggleApi();api.authenticate()
    write(OUT/'score-observer-launch.json',dict(started_utc=stamp(),pid=os.getpid(),deadline_utc=deadline.isoformat(),
        qualification_sha256=qsha,source_sha256=sha(__file__),maximum_accepted_entries=2,release_second_after_actual_first_score=True))
    def status(stage,**kw):
        row=dict(checked_utc=stamp(),pid=os.getpid(),stage=stage,deadline_utc=deadline.isoformat(),**kw)
        write(OUT/'score-observer-status.json',row);print('C42 ACTUAL SCORES',json.dumps(row),flush=True)
    while datetime.now(timezone.utc)<deadline:
        active=json.loads((ROOT/'.local/c38-active-goal.json').read_text());assert active['active'] and active['submissions_approved']==2
        assert sha(qpath)==qsha,'Reviewed qualification changed; no automatic release'
        ledger=json.loads(LEDGER.read_text());assert len(ledger['submissions'])<=2
        try:raw=api.competition_submissions(COMP)
        except Exception as error:
            status('actual_status_observation_failed',error_type=type(error).__name__,terminal_state_inferred=False)
            time.sleep(60);continue
        rows=[normalize(r) for r in raw];actual={r['reference']:r for r in rows};records={r['trial']:r for r in ledger['submissions']}
        missing=[]
        for record in ledger['submissions']:
            observed=actual.get(str(record['reference']))
            if observed is None:missing.append(str(record['reference']));continue
            newly_terminal=observed['status'] in TERMINAL and (record.get('status')!=observed['status'] or record.get('public_score')!=observed['public_score'])
            record.update(observed,actual_checked_utc=stamp())
            if newly_terminal:register_actual(record)
        write(LEDGER,ledger)
        if missing:
            status('accepted_reference_not_visible_yet',references=missing,terminal_state_inferred=False);time.sleep(60);continue
        cpu=records['cpu'];gpu=records.get('gpu')
        for trial,record in records.items():
            if record['status'] in ['ERROR','CANCELED','CANCELLED'] or record.get('error'):
                write(OUT/'actual-scoring-error.json',dict(checked_utc=stamp(),trial=trial,actual=record,automatic_retry=False,
                    accepted_entries=len(records),actual_scored_entries=sum(bool(r.get('success')) for r in records.values())))
                raise RuntimeError(trial+' actual scoring ended in error; inspect it without inventing a score or retrying an accepted trial')
        if cpu.get('success') and gpu is None:
            pending=[r['reference'] for r in rows if r['status'] not in TERMINAL]
            if pending:status('first_actual_score_observed_waiting_other_pending_entry',CPU=cpu['public_score'],pending=pending)
            elif sum(r.date.date()==datetime.now(timezone.utc).date() for r in raw)>=5:
                status('first_actual_score_observed_waiting_daily_quota',CPU=cpu['public_score'])
            else:
                status('first_actual_score_observed_releasing_frozen_GPU',CPU=cpu['public_score'],CPU_reference=cpu['reference'])
                submit('gpu');ledger=json.loads(LEDGER.read_text());assert len(ledger['submissions'])==2
                time.sleep(60);continue
        elif cpu.get('success') and gpu is not None and gpu.get('success'):
            assert len(records)==2 and all(r['status']=='COMPLETE' and r['public_score'] is not None for r in records.values())
            result=dict(completed_utc=stamp(),qualification_sha256=qsha,actual_scored_entries=2,entries=records,
                best_public_score_including_preserved_champion=max([.480]+[r['public_score'] for r in records.values()]),
                prior_0_point_480_artifacts_preserved=True,no_third_entry=True,C13_reused=False,observer_stopped=True)
            write(OUT/'actual-final-results.json',result);status('both_actual_scores_complete',**result);return
        else:status('waiting_actual_score',entries={t:dict(reference=r['reference'],status=r['status'],public_score=r.get('public_score')) for t,r in records.items()})
        time.sleep(60)
    raise TimeoutError('Bounded score observation expired; actual jobs may still be live, never infer termination or requeue')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--release-second',action='store_true');args=parser.parse_args()
    try:main(args.release_second)
    except BaseException as error:
        if OUT.exists():write(OUT/'score-observer-attention.json',dict(checked_utc=stamp(),error_type=type(error).__name__,reason=str(error),automatic_retry=False))
        raise
