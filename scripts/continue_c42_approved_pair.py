"""Advance this exact, already approved pair through its existing fixed gates.

This is a bounded continuation of the current job, not a recurring campaign.
It cannot fit, tune, change cohorts, requeue a notebook or choose another entry.
Every scoring mutation still goes through submit_c42_pair and its durable cap.
"""
import argparse,ctypes,hashlib,json,os,subprocess,sys,time
from ctypes import wintypes
from datetime import datetime,timedelta,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/local/c42-two-new-submissions-20261111'
C41=ROOT/'outputs/local/c41r3-masked-pipeline-validation-20261111'
C39=ROOT/'outputs/local/c42-fixed-c39-validation-20261111'
C43=ROOT/'outputs/local/c43-paired-confirmation-20261111'
FULL=ROOT/'outputs/forks/c41r3-full-masked-validation-v1'
CONSUMPTION=ROOT/'outputs/local/c13-confirmation-20261110/evaluation-consumption.json'
KERNEL='clarkkitchen/casmi26-c41r3-full-masked-validation'


def stamp():return datetime.now(timezone.utc).isoformat()


def read(path):return json.loads(path.read_text(encoding='utf8'))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2),encoding='utf8');temporary.replace(path)


def alive(pid):
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)];kernel.GetExitCodeProcess.restype=wintypes.BOOL
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    handle=kernel.OpenProcess(0x1000,False,int(pid))
    if not handle:return False
    try:
        code=wintypes.DWORD()
        return bool(kernel.GetExitCodeProcess(handle,ctypes.byref(code))) and code.value==259
    finally:kernel.CloseHandle(handle)


def admitted_pair(quality,independent):
    assert quality['queries']==352 and quality['all_queries_retained'] and quality['missing_truths_score_zero'] and not quality['C13_read']
    return independent['passed_pair'] and all(quality['results'][t]['admission']['passed'] and independent['results'][t]['passed'] for t in ['cpu','gpu'])


def main(release):
    assert release,'Explicit --release-approved-pair required for the existing two-entry authorization'
    import msvcrt
    with (OUT/'approved-pair-continuation.lock').open('a+b') as lock:
        lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        return continue_pair()


def continue_pair():
    from kaggle.api.kaggle_api_extended import KaggleApi
    launch=OUT/'approved-pair-continuation-launch.json'
    assert not launch.exists(),'Existing continuation must be reconciled, never blindly restarted'
    active=read(ROOT/'.local/c38-active-goal.json')
    assert active['active'] and active['submissions_approved']==2 and active['submissions_used']==0
    assert not active['hourly_automation'] and not active['new_paid_compute_authorized']
    policy=read(ROOT/'configs/c43-paired-confirmation-20261111.json')
    assert policy['automatic_confirmation_from_background_worker'] and policy['maximum_new_accepted_submissions']==2
    assert policy['submission_sequence']==['cpu','observe_actual_terminal_cpu_score','gpu','observe_actual_terminal_gpu_score']
    assert not CONSUMPTION.exists() and not (C43/'preparation.json').exists()
    assert not (OUT/'submission-qualification.json').exists() and not (ROOT/'.local/c42-two-new-submissions-20261111.json').exists()
    names=['continue_c42_approved_pair.py','watch_c41_validation.py','collect_c41_validation.py','audit_c39_acquisitions.py',
        'audit_c42_ranking.py','audit_c42_quality.py','evaluate_c42_fixed.py','confirm_c43_pair.py',
        'qualify_c42_pair.py','submit_c42_pair.py','watch_c42_scores.py','audit_c42_production.py','inspect_c42_remote.py']
    files=[ROOT/'scripts'/name for name in names]
    files += [ROOT/'configs'/name for name in ['c42-crossmodal-trials-20261111.json','c43-paired-confirmation-20261111.json']]
    files += [ROOT/'outputs/publish/c42r1-crossmodal-heads-20261111/head-manifest.json',
        ROOT/'outputs/local/c38-source-support-20261111/authorization.json',
        C41/'continuation-launch.json',OUT/'preview-worker-complete.json']
    preview=read(OUT/'preview-worker-complete.json')
    assert preview['leaderboard_submissions']==0 and not preview['C13_read']
    for trial in ['cpu','gpu']:
        path=ROOT/preview['audits'][trial]['path'];assert sha(path)==preview['audits'][trial]['sha256']
        audit=read(path);assert audit['execution_qualified'] and audit['local_cloud_final_lists_exact'] and audit['trial']==trial
        files.append(path)
        for suffix in ['notebook.ipynb','kernel-metadata.json','production-pin.json']:
            files.append(ROOT/'kaggle/forks'/('p20-c42-cpu-alignment' if trial=='cpu' else 'p21-c42-gpu-alignment')/suffix)
    sealed={str(p.relative_to(ROOT)):sha(p) for p in files}
    deadline=datetime.now(timezone.utc)+timedelta(hours=24)
    write(launch,dict(started_utc=stamp(),pid=os.getpid(),deadline_utc=deadline.isoformat(),source_seal=sealed,
        approved_new_entries=2,first_actual_score_before_second=True,one_paired_C13_only_after_fixed_C39_pass=True,
        no_new_recipe_or_cohort=True,no_new_cloud_run=True,no_paid_compute=True,native_goal_completion_requires_root_audit=True))
    def check():
        auth=read(ROOT/'.local/c38-active-goal.json')
        assert auth['active'] and auth['submissions_approved']==2 and auth['submissions_used']<=2
        assert datetime.now(timezone.utc)<deadline,'Observation deadline is not job termination; reconcile existing handles'
        for name,value in sealed.items():assert sha(ROOT/name)==value,name
    def status(stage,**extra):
        value=dict(checked_utc=stamp(),pid=os.getpid(),stage=stage,deadline_utc=deadline.isoformat(),**extra)
        write(OUT/'approved-pair-continuation-status.json',value);print('C42 APPROVED CONTINUATION',json.dumps(value),flush=True)
    def run(name,environment,script,*args):
        check();status(name)
        intent=OUT/(name+'-intent.json');assert not intent.exists(),'Unreconciled step intent: '+name
        command=[str(ROOT/environment/'Scripts/python.exe'),str(ROOT/'scripts'/script),*map(str,args)]
        write(intent,dict(started_utc=stamp(),command=command,source_seal_sha256=sha(launch),automatic_retry=False))
        with (OUT/(name+'.log')).open('w',encoding='utf8') as stream:
            result=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                timeout=max(1,(deadline-datetime.now(timezone.utc)).total_seconds()))
        assert result.returncode==0,name+' failed; inspect retained log without a new hypothesis or submission retry'
        check();write(intent,dict(**read(intent),completed_utc=stamp(),returncode=0))
    api=KaggleApi();api.authenticate()
    worker=read(C41/'continuation-launch.json')
    while not (C41/'continuation-complete.json').exists():
        check()
        try:state=json.loads(str(api.kernels_status(KERNEL)))
        except Exception as error:
            status('same_job_observation_unavailable',error_type=type(error).__name__,terminal_state_inferred=False)
            time.sleep(60);continue
        assert state['status'] not in ['ERROR','CANCELLED','CANCELED','CANCEL_ACKNOWLEDGED'],state
        if (C41/'continuation-attention.json').exists():raise RuntimeError('Existing validation observer needs investigation; do not requeue its cloud run')
        assert alive(worker['pid']) or (C41/'continuation-complete.json').exists(),'Existing validation observer exited; reconcile its exact job rather than requeue'
        status('waiting_same_full_validation_and_fixed_evaluation',kernel=KERNEL,cloud_status=state['status'],validation_observer_pid=worker['pid'])
        time.sleep(60)
    check();assert json.loads(str(api.kernels_status(KERNEL)))['status']=='COMPLETE'
    completed=read(C41/'continuation-complete.json');assert not completed['C13_read'] and completed['scoring_submissions']==0
    assert completed['quality_results_sha256']==sha(C39/'quality-results.json')
    return advance_pair(check,run,status)


def advance_pair(check,run,status):
    check()
    if not (C39/'independent-ranking-audit.json').exists():
        run('full-independent-ranking','.venv-gpu','audit_c42_ranking.py',C39,FULL/'c41_p01.csv')
    if not (C39/'independent-quality-audit.json').exists():
        run('full-independent-quality','.venv-gpu','audit_c42_quality.py')
    quality=read(C39/'quality-results.json');independent=read(C39/'independent-quality-audit.json')
    assert independent['quality_results_sha256']==sha(C39/'quality-results.json') and independent['source_sha256']==sha(ROOT/'scripts/audit_c42_quality.py')
    if not admitted_pair(quality,independent):
        value=dict(checked_utc=stamp(),stage='fixed_C39_gate_failed',quality_sha256=sha(C39/'quality-results.json'),
            independent_sha256=sha(C39/'independent-quality-audit.json'),C13_read=False,scoring_entries=0,
            admissions={trial:quality['results'][trial]['admission'] for trial in ['cpu','gpu']},no_automatic_recipe_or_cohort_change=True)
        write(OUT/'approved-pair-no-go.json',value);status('fixed_C39_gate_failed_stopped',decision=value);return
    run('paired-confirmation-prepare','.venv-gpu','confirm_c43_pair.py','prepare')
    run('paired-confirmation-score','.venv-gpu','confirm_c43_pair.py','score')
    run('paired-confirmation-consume-once','.venv-gpu','confirm_c43_pair.py','evaluate','--consume')
    confirmation=read(C43/'confirmation-results.json')
    if not all(confirmation['results'][trial]['admission']['passed'] for trial in ['cpu','gpu']):
        value=dict(checked_utc=stamp(),stage='single_paired_confirmation_failed',confirmation_sha256=sha(C43/'confirmation-results.json'),
            C13_joint_evaluations=1,scoring_entries=0,admissions={trial:confirmation['results'][trial]['admission'] for trial in ['cpu','gpu']},
            no_automatic_recipe_or_cohort_change=True)
        write(OUT/'approved-pair-no-go.json',value);status('single_paired_confirmation_failed_stopped',decision=value);return
    run('qualify-frozen-pair','.venv-gpu','qualify_c42_pair.py')
    run('first-approved-entry-preflight','.venv','submit_c42_pair.py','cpu','--dry-run')
    run('first-approved-entry-release','.venv','submit_c42_pair.py','cpu')
    run('observe-actual-first-release-second-stop-after-two','.venv','watch_c42_scores.py','--release-second')
    final=read(OUT/'actual-final-results.json')
    assert final['actual_scored_entries']==2 and final['no_third_entry'] and final['observer_stopped']
    assert all(row['status']=='COMPLETE' and row['success'] and row['public_score'] is not None for row in final['entries'].values())
    write(OUT/'approved-pair-continuation-complete.json',dict(completed_utc=stamp(),actual_results_sha256=sha(OUT/'actual-final-results.json'),
        actual_scored_entries=2,native_goal_marked_complete=False,root_actual_account_and_scope_audit_required=True))
    status('both_actual_results_ready_for_root_completion_audit',actual_scored_entries=2)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--release-approved-pair',action='store_true');args=parser.parse_args()
    try:main(args.release_approved_pair)
    except BaseException as error:
        if OUT.exists():write(OUT/'approved-pair-continuation-attention.json',dict(checked_utc=stamp(),error_type=type(error).__name__,reason=str(error),
            automatic_retry=False,observation_timeout_is_not_job_termination=True))
        raise
