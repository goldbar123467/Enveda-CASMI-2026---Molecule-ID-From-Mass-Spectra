"""Finish the current C58 source check after the already-running C57 audit.

One bounded current-task continuation. No fitting, cloud push, C13 consumption,
scoring entry or alternate recipe is possible here. Any failure stops for audit.
"""
import hashlib,json,os,subprocess,time
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/local/c58-complete-target-20261112'
BASE=ROOT/'outputs/local/c57-target-validation-20261112'
MODEL=ROOT/'outputs/local/c56-bounded-semantic-heads-20261112'
BUNDLE=ROOT/'outputs/publish/c56-bounded-semantic-heads-20261112'


def stamp():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as stream:
        while chunk:=stream.read(4*1024*1024):h.update(chunk)
    return h.hexdigest()
def write(p,v):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2),encoding='utf8');tmp.replace(p)


def main():
    import msvcrt
    with (OUT/'source-health-once.lock').open('a+b') as lock:
        lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        assert not (OUT/'source-health-once-launch.json').exists()
        deadline=datetime.fromisoformat(read(BASE/'base-only-launch.json')['deadline_utc'])
        sources=['audit_complete_core_health.py','continue_c58_source_once.py','c58_complete_target.py','evaluate_c58_target.py','audit_c58_quality.py','audit_c56_ranking.py']
        paths=[ROOT/'scripts'/s for s in sources]+[MODEL/'complete-candidate-freeze.json',MODEL/'independent-used-quality-audit.json',
            ROOT/'configs/c56-bounded-semantic-pair-20261112.json',OUT/'target-masked-reference/mask-manifest.json',OUT/'target-masked-reference/independent-mask-audit.json',BUNDLE/'head-manifest.json']
        manifest=read(BUNDLE/'head-manifest.json')
        for name,pin in manifest['files'].items():assert sha(BUNDLE/name)==pin;paths.append(BUNDLE/name)
        sealed={str(p.relative_to(ROOT)):sha(p) for p in paths}
        def check():
            active=read(ROOT/'.local/c38-active-goal.json');assert active['active'] and active['submissions_approved']==2 and active['submissions_used']==0
            assert not (ROOT/'outputs/local/c13-confirmation-20261110/evaluation-consumption.json').exists()
            for name,pin in sealed.items():assert sha(ROOT/name)==pin,name
        def status(stage,**kw):
            value=dict(checked_utc=stamp(),pid=os.getpid(),stage=stage,deadline_utc=deadline.isoformat(),**kw)
            write(OUT/'source-health-once-status.json',value);print('C58 SOURCE ONCE',json.dumps(value),flush=True)
        check();write(OUT/'source-health-once-launch.json',dict(started_utc=stamp(),pid=os.getpid(),source_seal=sealed,deadline_utc=deadline.isoformat(),
            can_fit=False,can_push=False,can_read_C13=False,can_submit=False,can_evaluate_only_frozen_full_C53_pair=True))
        status('waiting_existing_C57_collection_and_fidelity')
        while not (BASE/'base-only-complete.json').exists():
            check();assert not (BASE/'base-only-attention.json').exists(),'Inspect same C57 handle / collector; no retry'
            assert datetime.now(timezone.utc)<deadline,'C57 observer deadline; never infer termination or requeue'
            time.sleep(30)
        completed=read(BASE/'base-only-complete.json');cloud=ROOT/'outputs/forks/c57-full-masked-validation-v1'
        assert completed['collection_sha256']==sha(cloud/'collection-audit.json') and completed['fidelity_sha256']==sha(cloud/'acquisition-fidelity-audit.json')
        assert read(cloud/'acquisition-fidelity-audit.json')['all_original_acquisitions_retained']
        py=str(ROOT/'.venv-gpu/Scripts/python.exe');encoder=str(ROOT/'.venv-encoders/Scripts/python.exe')
        def run(stage,command):
            check();status(stage)
            with (OUT/(stage+'.log')).open('w',encoding='utf8') as log:
                subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        run('independent_complete_core_execution_audit',[py,'scripts/audit_complete_core_health.py',str(cloud),'--queries','482'])
        run('complete_P16_replay',[py,'scripts/c58_complete_target.py','replay'])
        run('both_actual_fresh_head_predictions',[encoder,str(BUNDLE/'score_c56_bounded_semantic.py'),str(BUNDLE),
            str(ROOT/'outputs/publish/frozen-p14-20261107'),str(OUT/'P16.csv'),str(OUT/'queries.parquet'),str(OUT/'head-predictions')])
        run('independent_ranking_audit',[py,'scripts/audit_c56_ranking.py',str(OUT/'head-predictions'),str(OUT/'P16.csv')])
        run('freeze_both_complete_source_predictions',[py,'scripts/evaluate_c58_target.py','freeze'])
        run('single_fixed_full_source_evaluation',[py,'scripts/evaluate_c58_target.py','evaluate'])
        run('independent_full_source_quality_audit',[py,'scripts/audit_c58_quality.py'])
        quality=read(OUT/'source-results.json');audit=read(OUT/'independent-quality-audit.json')
        assert audit['quality_results_sha256']==sha(OUT/'source-results.json') and quality['passed_pair']==audit['passed_pair']
        write(OUT/'source-health-once-complete.json',dict(completed_utc=stamp(),passed_pair=quality['passed_pair'],source_results_sha256=sha(OUT/'source-results.json'),
            independent_audit_sha256=sha(OUT/'independent-quality-audit.json'),C13_read=False,scoring_entries=0,
            next_action='Inspect actual full-panel admission. Only a passing pair may proceed to faithful C13 inference and once-only confirmation.'))
        status('fixed_source_evaluation_complete',passed_pair=quality['passed_pair'],C13_read=False,scoring_entries=0)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(OUT/'source-health-once-attention.json',dict(checked_utc=stamp(),error_type=type(error).__name__,reason=str(error),automatic_retry=False,
            C13_read=False,scoring_entries=0));raise
