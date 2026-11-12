"""Replay the original complete USED GL input to test batch/context effects.

Original model/code/input and numerical versions; no new training or quality.
One owned scope, host reserve4GiB, no automatic restart or external app stops.
"""
import os,json,subprocess,time
from datetime import datetime,timezone,timedelta
from c75_port_utils import ROOT,read,write,sha,stamp
from supervise_c75_local_port import authorization,ram,trim_own_cache


def main():
    authorization();base=ROOT/'outputs/local/c75-local-core-port-20261112'
    out=base/'gl-full-context-probe-v1';assert not out.exists();out.mkdir()
    numerical=read(base/'matched-numeric-runtime-probe.json')
    assert numerical['versions']['numpy']=='2.0.2' and numerical['versions']['numba']=='0.60.0'
    source=base/'original-cloud-intermediates/gl_work/gl_in.json'
    original_meta=read(base/'original-cloud-intermediates/gl_work/gl_out.json.meta.json')
    assert len(read(source))==original_meta['n_mols']==original_meta['n_mols_covered']==433
    assert read(base/'gl-exact-numeric-probe-v1/results.json')['comparison_to_original_T4']['outside_frozen_tolerance']==101
    assert not (ROOT/'outputs/local/c72-clean-target-holdout-20261112/evaluation-consumption.json').exists()
    assert not (ROOT/'outputs/local/c13-confirmation-20261110/evaluation-consumption.json').exists()
    assert ram()>=12
    winroot=str(ROOT);assert winroot.startswith('C:')
    linux='/mnt/c/Users/imsa/Documents/Enveda CASMI/outputs/local/c75-local-core-port-20261112'
    package=linux+'/input/casmi26-attributed-v4g-reproduction-assets/casmi26-glacier'
    runner=base/'input/casmi26-attributed-v4g-reproduction-assets/casmi26-glacier/gl_runner.py'
    assert sha(runner)==read(base/'gl-engineering-probe-v1/intent.json')['runner_sha256']
    unit='enveda-c75-gl-full-context-'+str(os.getpid())
    wsl=['wsl.exe','-d','Ubuntu-24.04','--']
    command=wsl+['systemd-run','--user','--scope','--unit='+unit,
        '-p','MemoryHigh=6G','-p','MemoryMax=8G','-p','MemorySwapMax=2G',
        '/usr/bin/env','OMP_NUM_THREADS=2','MKL_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2','NUMBA_NUM_THREADS=2','PYTHONHASHSEED=0','CUDA_VISIBLE_DEVICES=0',
        '/home/imsa/.venvs/enveda-c75-torch210/bin/python',package+'/gl_runner.py',package,
        linux+'/original-cloud-intermediates/gl_work/gl_in.json',linux+'/gl-full-context-probe-v1/gl_out.json',
        '--device','cuda','--budget','4000.0','--site','/home/imsa/.local/share/enveda-c75/fixture/working/ice_site',
        '--wheels',linux+'/input/casmi26-attributed-v4g-reproduction-assets/casmi26-iceberg/wheels','--any-rdkit']
    deadline=datetime.now(timezone.utc)+timedelta(minutes=75)
    write(out/'intent.json',dict(started_utc=stamp(),pid=os.getpid(),source_sha256=sha(__file__),unit=unit,
        input_sha256=sha(source),runner_sha256=sha(runner),runtime_probe_sha256=sha(base/'matched-numeric-runtime-probe.json'),
        original_cloud_raw_output_sha256=sha(base/'original-cloud-raw-forward/gl_work/gl_out.json'),
        whole_original_input=True,original_molecules=433,covered_molecules_expected=433,
        original_args_and_budget=True,host_reserve_GiB=4,task_memory_high_GiB=6,task_memory_max_GiB=8,task_swap_max_GiB=2,
        deadline_utc=deadline.isoformat(),exactly_one_launch=True,USED_only=True,new_source_quality_read=False,C13_read=False,scoring_entries=0))
    with (out/'replay.log').open('w',encoding='utf8') as stream:
        child=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        try:
            while child.poll() is None:
                authorization();assert datetime.now(timezone.utc)<deadline
                free=ram()
                if free<4:
                    trim_own_cache();time.sleep(5);free=ram();assert free>=4,('Host reserve below4GiB',free)
                write(out/'status.json',dict(checked_utc=stamp(),pid=os.getpid(),child_windows_pid=child.pid,unit=unit,
                    phase='original_complete_GL_context_replay_running',host_free_GiB=free,launches=1,
                    USED_only=True,new_source_quality_read=False,C13_read=False,scoring_entries=0))
                time.sleep(10)
            assert child.returncode==0,('Original GL replay failed; preserve evidence',child.returncode)
            meta=read(out/'gl_out.json.meta.json')
            assert meta['status']=='ok' and meta['n_mols_scored']==meta['n_mols_covered']==meta['n_mols']==433
            assert meta['n_pred_fail']==meta['n_feat_fail']==0
        except BaseException as exc:
            if child.poll() is None:
                # Scope name is unique to this recorded parent PID. Verify its
                # cgroup before stopping this task only; never stop WSL/apps.
                r=subprocess.run(wsl+['systemctl','--user','show',unit,'-p','ControlGroup','--value'],capture_output=True,text=True,check=True)
                assert r.stdout.strip().endswith('/'+unit+'.scope'),r.stdout
                subprocess.run(wsl+['systemctl','--user','stop',unit+'.scope'],check=True,timeout=30)
                child.wait(timeout=30)
            write(out/'failed.json',dict(checked_utc=stamp(),error=type(exc).__name__+': '+str(exc)[:500],
                source_sha256=sha(__file__),automatic_retry=False,USED_only=True,new_source_quality_read=False,C13_read=False,scoring_entries=0))
            raise
    write(out/'complete.json',dict(completed_utc=stamp(),source_sha256=sha(__file__),intent_sha256=sha(out/'intent.json'),
        output_sha256=sha(out/'gl_out.json'),metadata_sha256=sha(out/'gl_out.json.meta.json'),
        all_original_433_supported_molecules_completed=True,no_prediction_or_feature_failures=True,
        original_cloud_raw_score_comparison_still_required=True,USED_only=True,new_source_quality_read=False,C13_read=False,scoring_entries=0))
    print('C75 FULL ORIGINAL GL CONTEXT COMPLETE; RAW SCORE PARITY AUDIT REQUIRED',flush=True)


if __name__=='__main__':main()
