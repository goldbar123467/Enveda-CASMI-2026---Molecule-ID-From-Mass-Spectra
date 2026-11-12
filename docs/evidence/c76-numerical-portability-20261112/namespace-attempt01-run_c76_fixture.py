"""Execute only the complete frozen C76 USED12 fixture; never fresh quality."""
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS']:
    os.environ[name]='2'
os.environ['MPLBACKEND']='Agg';os.environ['PYTHONUNBUFFERED']='1';os.environ['CUDA_VISIBLE_DEVICES']='0'
import fcntl,json,sys,time
from pathlib import Path
from c75_port_utils import ROOT,read,write,sha,stamp


def main():
    assert sys.prefix=='/home/imsa/.venvs/enveda-c75-torch210'
    base=ROOT/'outputs/local/c76-numerical-portability-20261112';out=base/'fixture-port'
    old=ROOT/'outputs/local/c75-local-core-port-20261112'
    lock=(old/'local-inference.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    own_lock=(base/'local-inference.lock').open('a+b');fcntl.flock(own_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    pin=read(out/'fixture-freeze.json');assert pin['notebook_sha256']==sha(out/'notebook.ipynb')
    audit=read(out/'independent-source-audit.json');assert audit['passed'] and audit['fixture_freeze_sha256']==sha(out/'fixture-freeze.json')
    assert audit['source_sha256']==sha(ROOT/'scripts/audit_c76_fixture_source.py')
    assert pin['source_sha256']==sha(ROOT/'scripts/prepare_c76_fixture.py')
    assert pin['queries']==12 and pin['acquisitions']==51 and pin['C75_gate_still_failed']
    assert not (out/'run-intent.json').exists()
    assert read(ROOT/'.local/c38-active-goal.json')['active']
    assert not (ROOT/'outputs/local/c72-clean-target-holdout-20261112/evaluation-consumption.json').exists()
    assert not (ROOT/'outputs/local/c13-confirmation-20261110/evaluation-consumption.json').exists()
    assert 'enveda-c76-fixture-' in Path('/proc/self/cgroup').read_text()
    working=Path(pin['owned_working']);pkg=Path(pin['owned_GL_package'])
    assert working==Path('/home/imsa/.local/share/enveda-c76/fixture/working') and not any(working.iterdir())
    assert sha(pkg/'gl_runner.py')==pin['candidate_gl_runner_sha256'] and sha(pkg/'c76_gl_portability.py')==pin['helper_sha256']
    assert sha(pkg/'ckpt/glacier.pt')==pin['original_checkpoint_sha256']
    input_link=Path('/kaggle/input');work_link=Path('/kaggle/working')
    assert input_link.is_symlink() and str(input_link.resolve())==pin['original_input_target']
    assert work_link.is_symlink() and work_link.resolve()==Path('/home/imsa/.local/share/enveda-c75/fixture/working')
    temp=Path('/kaggle/.c76-owned-working-link');assert not temp.exists() and not temp.is_symlink()
    temp.symlink_to(working,target_is_directory=True);os.replace(temp,work_link)
    assert work_link.resolve()==working
    os.environ['CASMI_C76_GL_PACKAGE']=str(pkg)
    site=out/'resource-site';site.mkdir()
    text="import os\nfor n in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[n]='2'\nos.cpu_count=lambda:4\n"
    (site/'sitecustomize.py').write_text(text,encoding='utf8')
    os.environ['PYTHONPATH']=str(site)+os.pathsep+os.environ.get('PYTHONPATH','');os.cpu_count=lambda:4
    if os.getpgrp()!=os.getpid():os.setsid()
    write(out/'namespace-activation.json',dict(activated_utc=stamp(),working_target=str(working),
        input_target=str(input_link.resolve()),owned_GL_package=str(pkg),only_owned_working_link_switched=True,old_data_preserved=True))
    write(out/'run-intent.json',dict(started_utc=stamp(),pid=os.getpid(),pgid=os.getpgrp(),source_sha256=sha(__file__),
        fixture_freeze_sha256=sha(out/'fixture-freeze.json'),candidate_freeze_sha256=sha(base/'candidate-freeze-r1.json'),
        actual_single_2070=True,actual_two_T4s=False,USED_only=True,quality_read=False,C13_read=False,scoring_entries=0))
    notebook=read(out/'notebook.ipynb');scope={'__name__':'__main__'};started=time.monotonic()
    try:
        for i,cell in enumerate(notebook['cells']):
            if cell['cell_type']!='code':continue
            write(out/'run-progress.json',dict(checked_utc=stamp(),pid=os.getpid(),outer_cell=i,phase='START',elapsed_seconds=time.monotonic()-started))
            print('C76 OUTER CELL',i,'START',flush=True)
            exec(compile(''.join(cell['source']),'c76_outer_cell_'+str(i),'exec'),scope)
            print('C76 OUTER CELL',i,'DONE',flush=True)
        report=read(working/'c41_report.json')
        assert report['queries']==12 and report['baseline_replay_exact'] and not report['metrics_computed'] and not report['query_labels_mounted']
        assert report['p14_receipt']['input_errors']==0
        write(out/'run-complete.json',dict(completed_utc=stamp(),source_sha256=sha(__file__),queries=12,acquisitions=51,
            elapsed_seconds=time.monotonic()-started,working=str(working),P01_sha256=sha(working/'c41_p01.csv'),P14_sha256=sha(working/'c41_p14.csv'),
            original_C75_gate_passed=False,independent_health_source_fidelity_and_GL_branch_audits_required=True,
            USED_only=True,quality_read=False,C13_read=False,scoring_entries=0,qualified_for_scoring=False))
    except BaseException as exc:
        write(out/'run-failed.json',dict(checked_utc=stamp(),source_sha256=sha(__file__),error=repr(exc),automatic_retry=False,
            USED_only=True,quality_read=False,C13_read=False,scoring_entries=0));raise
    print('C76 COMPLETE USED12 PIPELINE RETURNED; INDEPENDENT AUDITS STILL REQUIRED',flush=True)


if __name__=='__main__':main()
