"""Exercise the unchanged N1 proxy driver with the explicit C76 GL loader.

Synthetic adduct/CE variants of one previously frozen USED input, four existing
candidates, no truth and no quality metric. Run after the complete fixture exits.
"""
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[name]='2'
import argparse,copy,importlib.util,math,subprocess,sys
from pathlib import Path
from c75_port_utils import ROOT,read,write,sha,stamp


def paths():
    base=ROOT/'outputs/local/c76-numerical-portability-20261112'
    old=Path('/home/imsa/.local/share/enveda-c75/fixture/working/n1_work')
    assert sha(old/'glb_core.py')=='f7a3b8cd8802a7ba3b69a297603241784a63c069044cf7bd6efda7d68cc8fdfa'
    assert sha(old/'glb_driver.py')=='e919a154b57bda5a52573481348e38d8b4e7c97883532ea5b385bc3cb6f98502'
    return base,old


def prepare():
    base,old=paths();out=base/'n1-resource-path-probe';assert not out.exists();out.mkdir()
    spec=importlib.util.spec_from_file_location('c76_original_n1_core',old/'glb_core.py')
    core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
    src=ROOT/'outputs/local/c75-local-core-port-20261112/device-path-probe-v1/intent.json'
    first=read(src)['selected'][0]
    real=next(s for s in first['spectra'] if s['adduct']=='[M+H]+' and s['mode']=='positive')
    spectra=[]
    for adduct in ['[M+NH4]+','[M-H2O+H]+']:
        for merged in [False,True]:
            sp=copy.deepcopy(real)
            precursor=float(real['prec'])+(core.NH4_ADD-core.PROTON if adduct=='[M+NH4]+' else -core.H2O)
            sp.update(adduct=adduct,prec=precursor,ce_ev=[20.,40.] if merged else [40.],ce_orig='20,40' if merged else '40',ce_units='eV')
            sp['mz']=list(sp['mz'])+[precursor];sp['it']=list(sp['it'])+[1.]
            proxy=core.n1_proxy(sp);assert proxy is not None and len(proxy['mz'])<len(sp['mz'])
            spectra.append(sp)
    items,counts=core.build_variants([dict(mid='C76_SYNTHETIC_N1_PATH',cands=first['cands'],spectra=spectra)])
    assert counts=={'NH4p':2,'WLp':2} and len(items[0]['cands'])==4
    write(out/'input.json',items)
    fixture=read(base/'fixture-resource-port/fixture-freeze.json')
    write(out/'intent.json',dict(started_utc=stamp(),source_sha256=sha(__file__),original_fixed_cases_sha256=sha(src),
        input_sha256=sha(out/'input.json'),original_N1_core_sha256=sha(old/'glb_core.py'),original_N1_driver_sha256=sha(old/'glb_driver.py'),
        candidate_GL_runner_sha256=fixture['candidate_gl_runner_sha256'],helper_sha256=fixture['helper_sha256'],
        synthetic_only=True,candidates=4,proxy_spectra=4,merged_CE_spectra=2,tags=counts,
        selection='First already frozen16case input; all its4candidates. Synthetic NH4/water-loss plus single/merged CE path exercise, never quality.',
        acceptance='Real CPU model; all4spectra covered; merged2; all32mean/merged candidate outputs finite; no prediction/feature errors',
        USED_only=True,quality_read=False,C13_read=False,scoring_entries=0))
    print('C76 SYNTHETIC N1 PATH INPUT FROZEN; NO INFERENCE',flush=True)


def run():
    base,old=paths();out=base/'n1-resource-path-probe';pin=read(out/'intent.json')
    assert pin['source_sha256']==sha(__file__) and pin['input_sha256']==sha(out/'input.json')
    assert (base/'fixture-resource-port/run-complete.json').exists(),'Do not overlap the complete fixture with extra ML work'
    assert not (out/'run-intent.json').exists();assert not (out/'output.json').exists()
    fixture=read(base/'fixture-resource-port/fixture-freeze.json');pkg=Path(fixture['owned_GL_package'])
    assert sha(pkg/'gl_runner.py')==pin['candidate_GL_runner_sha256']
    assert 'enveda-c76-resource-n1-path' in Path('/proc/self/cgroup').read_text()
    write(out/'run-intent.json',dict(started_utc=stamp(),source_sha256=sha(__file__),input_intent_sha256=sha(out/'intent.json'),real_CPU_only=True,
        USED_only=True,quality_read=False,C13_read=False,scoring_entries=0))
    command=[sys.executable,str(old/'glb_driver.py'),str(pkg),str(out/'input.json'),str(out/'output.json'),
        '--device','cpu','--budget','400','--site','/home/imsa/.local/share/enveda-c75/fixture/working/ice_site',
        '--wheels',str(ROOT/'outputs/local/c75-local-core-port-20261112/input/casmi26-attributed-v4g-reproduction-assets/casmi26-iceberg/wheels')]
    with (out/'model.log').open('w',encoding='utf8') as stream:result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=500)
    assert result.returncode==0
    meta=read(out/'output.json.meta.json');assert meta['status']=='ok' and not meta['fake'] and meta['device']=='cpu'
    assert meta['n_mols_done']==1 and meta['n_spec_covered']==4 and meta['n_spec_merged']==2
    assert meta['n_pred_fail']==meta['n_feat_fail']==0
    adapter=read(out/'output.json.c76_adapter.json');assert adapter['weights_exact'] and adapter['decision_device']==adapter['intensity_device']=='cpu'
    data=read(out/'output.json')['C76_SYNTHETIC_N1_PATH'];values=[]
    for spectrum in data['spec'].values():
        for mode in ['mean','merged']:values+=spectrum[mode]
    assert len(values)==32 and all(isinstance(v,(float,int)) and math.isfinite(v) for v in values)
    write(out/'results.json',dict(completed_utc=stamp(),source_sha256=sha(__file__),input_intent_sha256=sha(out/'intent.json'),
        original_driver_real_model=True,proxy_spectra_covered=4,merged_CE_spectra=2,finite_outputs=32,
        prediction_or_feature_errors=0,C76_loader_executed=True,output_sha256=sha(out/'output.json'),metadata_sha256=sha(out/'output.json.meta.json'),
        adapter_receipt_sha256=sha(out/'output.json.c76_adapter.json'),passed=True,synthetic_only=True,
        USED_only=True,quality_read=False,C13_read=False,scoring_entries=0))
    print('C76 REAL N1 CPU PROXY/CE/LOADER PATH PASSED',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','run']);args=parser.parse_args()
    (prepare if args.phase=='prepare' else run)()
