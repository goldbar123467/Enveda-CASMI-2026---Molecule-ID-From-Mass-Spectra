"""One complete actual production preview, followed by both unchanged heads."""
import os
for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS']: os.environ[name] = '2'
os.environ.update(MPLBACKEND='Agg', PYTHONUNBUFFERED='1', PYTHONHASHSEED='0', CUDA_VISIBLE_DEVICES='0', PYTHONDONTWRITEBYTECODE='1')
import fcntl
import shutil
import sys
import time
from pathlib import Path
from c75_port_utils import ROOT, read, write, sha, stamp


def main():
    assert sys.prefix == '/home/imsa/.venvs/enveda-c75-torch210'
    base = ROOT / 'outputs/local/c76-numerical-portability-20261112'; out = base / 'local-production-preview'
    assert 'enveda-c76-production-preview-' in Path('/proc/self/cgroup').read_text()
    locks = []
    for path in [ROOT / 'outputs/local/c75-local-core-port-20261112/local-inference.lock', base / 'local-inference.lock']:
        handle = path.open('a+b'); fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB); locks.append(handle)
    plan = read(out / 'launch-plan.json')
    for name, digest in plan['files_sha256'].items(): assert sha(ROOT / name) == digest, name
    assert read(out / 'independent-source-audit.json')['passed']
    pin = read(out / 'preview-freeze.json'); working = Path(pin['working'])
    assert Path('/kaggle/input').resolve() == Path(pin['input']) and Path('/kaggle/working').resolve() == working
    assert not any(working.iterdir()) and not (out / 'run-intent.json').exists()
    assert read(out / 'namespace-activation.json')['launch_plan_sha256'] == sha(out / 'launch-plan.json')
    assert read(ROOT / '.local/c38-active-goal.json')['active']
    for name in ['c72-clean-target-holdout-20261112', 'c13-confirmation-20261110']:
        assert not (ROOT / 'outputs/local' / name / 'evaluation-consumption.json').exists()
    for name in ['KAGGLE_IS_COMPETITION_RERUN', 'CASMI_FORCE_RERUN', 'CASMI_C41_SID_MASK', 'CASMI_C41_EXCLUSION_FILE', 'CASMI_C08_VALIDATION_COMP']:
        assert not os.environ.get(name), ('Unexpected validation/rerun environment', name)
    site = out / 'resource-site'; site.mkdir()
    (site / 'sitecustomize.py').write_text("import os\nfor n in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[n]='2'\nos.cpu_count=lambda:4\n", encoding='utf8')
    os.environ['PYTHONPATH'] = str(site) + os.pathsep + os.environ.get('PYTHONPATH', '')
    os.cpu_count = lambda: 4
    if os.getpgrp() != os.getpid(): os.setsid()
    started = time.monotonic(); source_pin = sha(__file__)
    write(out / 'run-intent.json', dict(started_utc=stamp(), pid=os.getpid(), pgid=os.getpgrp(),
        source_sha256=source_pin, launch_plan_sha256=sha(out / 'launch-plan.json'),
        preview_freeze_sha256=sha(out / 'preview-freeze.json'), actual_single_2070=True, actual_two_T4s=False,
        preview_queries=12, preview_acquisitions=pin['preview_acquisitions'], output_rows=400,
        no_new_head_fitting=True, quality_read=False, C13_read=False, scoring_entries=0))
    scope = {'__name__': '__main__'}
    try:
        for i, cell in enumerate(read(out / 'notebook.ipynb')['cells']):
            if cell['cell_type'] != 'code': continue
            write(out / 'run-progress.json', dict(checked_utc=stamp(), pid=os.getpid(), outer_cell=i,
                phase='START', elapsed_seconds=time.monotonic()-started))
            print('C76 PRODUCTION OUTER CELL', i, 'START', flush=True)
            exec(compile(''.join(cell['source']), 'c76_local_production_' + str(i), 'exec'), scope)
            if i == 6: assert not scope['IS_RERUN'] and len(scope['SAMPLE_IDS']) == 400
            print('C76 PRODUCTION OUTER CELL', i, 'DONE', flush=True)
        assert not scope['IS_RERUN']
        shutil.copyfile(working / 'submission.csv', working / 'c76_cpu_submission.csv')
        shutil.copyfile(working / 'c76_p16.csv', working / 'submission.csv')
        write(out / 'run-progress.json', dict(checked_utc=stamp(), pid=os.getpid(), phase='GPU_HEAD', elapsed_seconds=time.monotonic()-started))
        exec(compile((out / 'gpu-head-stage.py').read_text(encoding='utf8'), 'c76_local_gpu_head', 'exec'), scope)
        shutil.copyfile(working / 'submission.csv', working / 'c76_gpu_submission.csv')
        for trial, receipt_name in [('cpu', 'c76_execution.json'), ('gpu', 'c76_gpu_execution.json')]:
            receipt = read(working / receipt_name)
            assert receipt['queries'] == 12 and receipt['acquisitions'] == pin['preview_acquisitions']
            assert receipt['valid_rows'] == 400 and receipt['exact_unprocessed_fallback_rows'] == 388
            assert receipt['trials'][trial]['changed_lists'] > 0 and not receipt['is_hidden_rerun']
        assert sha(__file__) == source_pin
        write(out / 'run-complete.json', dict(completed_utc=stamp(), source_sha256=source_pin,
            elapsed_seconds=time.monotonic()-started, preview_queries=12, preview_acquisitions=pin['preview_acquisitions'],
            output_rows=400, inactive_rows=388, working=str(working),
            CPU_csv_sha256=sha(working / 'c76_cpu_submission.csv'), GPU_csv_sha256=sha(working / 'c76_gpu_submission.csv'),
            actual_single_2070=True, actual_T4_evidence=False, independent_full_audit_still_required=True,
            quality_read=False, C13_read=False, scoring_entries=0, qualified_for_scoring=False))
    except BaseException as exc:
        (working / 'submission.csv').unlink(missing_ok=True)
        write(out / 'run-failed.json', dict(checked_utc=stamp(), source_sha256=source_pin, error=repr(exc),
            automatic_retry=False, quality_read=False, C13_read=False, scoring_entries=0)); raise
    print('C76 REAL LOCAL PRODUCTION PREVIEW AND BOTH HEADS FINISHED; INDEPENDENT AUDIT REQUIRED', flush=True)


if __name__ == '__main__': main()
