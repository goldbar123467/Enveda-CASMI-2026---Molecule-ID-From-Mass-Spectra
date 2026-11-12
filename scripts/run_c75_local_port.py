"""Execute the audited complete local port; never read quality labels or score."""
import os
for n in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[n]='2'
os.environ['MPLBACKEND']='Agg'
os.environ['PYTHONUNBUFFERED']='1'
os.environ['PYTHONHASHSEED']='0'
os.environ['CUDA_VISIBLE_DEVICES']='0'
import argparse,fcntl,json,platform,sys,time
from pathlib import Path
from c75_port_utils import ROOT,read,write,sha,stamp
OUT=ROOT/'outputs/local/c75-local-core-port-20261112'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['fixture','full']);args=parser.parse_args();mode=args.mode
    assert platform.system()=='Linux' and sys.prefix=='/home/imsa/.venvs/enveda-c75'
    lock=(OUT/'local-inference.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    target=OUT/(mode+'-port');pin=read(target/'port-freeze.json');audit=read(target/'port-source-audit.json')
    assert audit['source_and_fixture_audit_passed'] and audit['port_freeze_sha256']==sha(target/'port-freeze.json')
    assert pin['local_notebook_sha256']==sha(target/'notebook.ipynb')
    assert pin['source_sha256']==sha(ROOT/'scripts/prepare_c75_port.py')
    assert not (target/'run-intent.json').exists(),'Reconcile the original PID/artifacts; no blind restart'
    auth=read(ROOT/'.local/c38-active-goal.json');assert auth['active'] and auth['free_local_and_kaggle_compute_authorized']
    assert auth['submissions_used']==0 and auth['scientific_gate_hold']
    assert not (ROOT/'outputs/local/c72-clean-target-holdout-20261112/evaluation-consumption.json').exists()
    assert not (ROOT/'outputs/local/c13-confirmation-20261110/evaluation-consumption.json').exists()
    if mode=='full':assert read(OUT/'fixture-parity-audit.json')['passed']
    v4g=read(OUT/'v4g-asset-freeze.json');aux=read(OUT/'aux-asset-freeze.json')
    assert v4g['files']==237 and len(v4g['files_sha256'])==237 and v4g['original_remote_census_exact']
    assert v4g['source_sha256']==sha(ROOT/'scripts/prepare_c75_v4g_assets.py')
    assert aux['source_sha256']==sha(ROOT/'scripts/prepare_c75_aux_assets.py') and len(aux['files_sha256'])==388
    for name,size in {r['name']:r['bytes'] for r in read(ROOT/'outputs/research/c20-engine-recovery-20261110/asset-files.json')['files']}.items():
        assert (OUT/'input/casmi26-attributed-v4g-reproduction-assets'/name).stat().st_size==size,name
    census=read(OUT/'aux-asset-inventory.json')
    for dataset,files in census['datasets'].items():
        for name,size in files.items():assert (OUT/'input'/dataset.split('/')[1]/name).stat().st_size==size,(dataset,name)
    namespace=read(target/'namespace-activation.json');working=Path(namespace['working_target'])
    assert namespace['mode']==mode and namespace['only_owned_links_switched']
    assert Path('/kaggle/working').resolve()==working and Path('/kaggle/input').resolve()==Path(namespace['input_target'])
    assert not (working/'p2s_code').exists() and not (working/'execution_receipt.json').exists(),'Fresh owned working directory required'
    # A child-process sitecustomize matches the cloud's four CPU workers and
    # prevents oversubscription on the desktop. It changes no scientific code.
    site=target/'resource-site';site.mkdir(exist_ok=True)
    text="import os\nfor n in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[n]='2'\nos.cpu_count=lambda:4\n"
    (site/'sitecustomize.py').write_text(text,encoding='utf8')
    os.environ['PYTHONPATH']=str(site)+os.pathsep+os.environ.get('PYTHONPATH','');os.cpu_count=lambda:4
    if os.getpgrp()!=os.getpid():os.setsid()
    assert os.getpgrp()==os.getpid()
    write(target/'run-intent.json',dict(started_utc=stamp(),pid=os.getpid(),pgid=os.getpgrp(),source_sha256=sha(__file__),mode=mode,
        port_freeze_sha256=sha(target/'port-freeze.json'),port_source_audit_sha256=sha(target/'port-source-audit.json'),
        asset_freezes_sha256=[sha(OUT/'v4g-asset-freeze.json'),sha(OUT/'aux-asset-freeze.json')],
        operational_CPU_workers=4,threads_per_process=2,resource_site_sha256=sha(site/'sitecustomize.py'),
        quality_outcomes_read=False,C13_read=False,scoring_entries=0,actual_two_T4s=False))
    nb=read(target/'notebook.ipynb');scope={'__name__':'__main__'};started=time.monotonic()
    try:
        for i,cell in enumerate(nb['cells']):
            if cell['cell_type']!='code':continue
            write(target/'run-progress.json',dict(checked_utc=stamp(),pid=os.getpid(),pgid=os.getpgrp(),outer_cell=i,stage='START',elapsed_seconds=time.monotonic()-started,quality_outcomes_read=False,C13_read=False))
            print('C75 OUTER CELL',i,'START',flush=True)
            exec(compile(''.join(cell['source']),'c75_outer_cell_'+str(i),'exec'),scope)
            print('C75 OUTER CELL',i,'DONE',flush=True)
        report=read(working/'c41_report.json')
        assert report['queries']==pin['queries'] and report['baseline_replay_exact'] and not report['metrics_computed'] and not report['query_labels_mounted']
        assert report['p14_receipt']['input_errors']==0
        write(target/'run-complete.json',dict(completed_utc=stamp(),source_sha256=sha(__file__),mode=mode,queries=pin['queries'],acquisitions=pin['acquisitions'],
            elapsed_seconds=time.monotonic()-started,working=str(working),P01_sha256=sha(working/'c41_p01.csv'),P14_sha256=sha(working/'c41_p14.csv'),
            actual_local_single_2070=True,actual_two_T4s=False,quality_outcomes_read=False,C13_read=False,scoring_entries=0,
            independent_collection_health_fidelity_and_parity_audits_still_required=True))
    except BaseException as error:
        write(target/'run-failed.json',dict(checked_utc=stamp(),pid=os.getpid(),pgid=os.getpgrp(),error=type(error).__name__+': '+str(error)[:500],
            mode=mode,automatic_retry=False,quality_outcomes_read=False,C13_read=False,scoring_entries=0))
        raise
    print('C75 LOCAL COMPLETE PIPELINE RETURNED',mode,'independent audits still required',flush=True)


if __name__=='__main__':main()
