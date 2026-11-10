"""Exercise the staged production overlay on immutable, previously used inputs."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
import json
from pathlib import Path
import pyarrow as pa
from precision_cache import ROOT,features,write,sha
from build_c09_controls import unwrap
from build_c13_formate_recovery import value
from public_reference_fixture import run
from training_monitor import system_memory


def main():
    pa.set_cpu_count(2);assert system_memory()['free_gib']>=4
    out=ROOT/'outputs/local/c33-public-production-20261110';input_root=out/'fixture-input';input_root.mkdir(exist_ok=True)
    ext=input_root/'extlib';ext.mkdir(exist_ok=True);mb=input_root/'massbank';mb.mkdir(exist_ok=True)
    cache=ROOT/'outputs/research/2026-11-09-silver'
    for name in ['ext_meta.parquet','ext_mz.npy','ext_in.npy','ext_mass.parquet']:
        dest=ext/name;src=cache/'extlib'/name
        if not dest.exists():os.link(src,dest)
        assert sha(dest)==sha(src)
    src=cache/'massbank-coverage-audit/spectra.parquet';dest=mb/'spectra.parquet'
    if not dest.exists():os.link(src,dest)
    assert sha(dest)==sha(src)
    package=ROOT/'outputs/publish/c09e-smoke-fixtures-20261109';test=input_root/'test.parquet'
    if not test.exists():os.link(package/'c09e_fixture_queries.parquet',test)
    nb=json.loads((ROOT/'kaggle/forks/p17-public-mass-dualref/notebook.ipynb').read_text(encoding='utf-8'))
    ending=unwrap(''.join(nb['cells'][-1]['source']));core=value(ending,'C33_PUBLIC_CORE');overlay=value(core,'C29_OVERLAY')
    work=out/'reference-fixture-local';assert not (work/'fixture-receipt.json').exists(),'Fixture already completed; reuse its pinned source or inspect before a changed-source rerun'
    result=run(work,input_root,package,overlay,features,value(ending,'C33_KNOWN_ADDUCTS'))
    write(out/'fixture-expectations.json',result)
    print('DONE C33 INDEPENDENT LOCAL REFERENCE FIXTURE',json.dumps(dict(real_queries=30,fallbacks=4,
        views={k:dict(changed_rows=v['changed_rows'],content_sha256=v['predictions_content_sha256']) for k,v in result['views'].items()},independent=result['independent'])),flush=True)


if __name__=='__main__':main()
