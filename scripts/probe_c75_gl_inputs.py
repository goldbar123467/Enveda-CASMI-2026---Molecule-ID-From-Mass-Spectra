"""Two small GL engineering replays on USED inputs; never quality or fitting."""
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS']:
    os.environ[name]='2'
os.environ['PYTHONHASHSEED']='0';os.environ['CUDA_VISIBLE_DEVICES']='0'
import subprocess,sys
from pathlib import Path
from c75_port_utils import ROOT,read,write,sha,stamp


def main():
    base=ROOT/'outputs/local/c75-local-core-port-20261112'
    out=base/'gl-engineering-probe-v1';assert not out.exists();out.mkdir()
    work=Path('/home/imsa/.local/share/enveda-c75/fixture/working')
    local_input=work/'gl_work/gl_in.json';local_output=work/'gl_work/gl_out.json'
    old_input=base/'original-cloud-intermediates/gl_work/gl_in.json'
    local=read(local_input);old=read(old_input);ids={item['mid'] for item in local}
    selected=[item for item in old if item['mid'] in ids]
    assert len(local)==len(selected)==9
    lookup={item['mid']:item for item in local}
    assert all(item['spectra']==lookup[item['mid']]['spectra'] for item in selected)
    write(out/'original-cloud-nine-input.json',selected)
    package=base/'input/casmi26-attributed-v4g-reproduction-assets/casmi26-glacier'
    runner=package/'gl_runner.py'
    scope=Path('/proc/self/cgroup').read_text().strip()
    assert 'enveda-c75-gl-probe' in scope
    write(out/'intent.json',dict(started_utc=stamp(),pid=os.getpid(),source_sha256=sha(__file__),
        runner_sha256=sha(runner),local_input_sha256=sha(local_input),local_original_output_sha256=sha(local_output),
        cloud_input_sha256=sha(old_input),cloud_nine_input_sha256=sha(out/'original-cloud-nine-input.json'),
        experiments=['repeat_identical_local_input','original_cloud_nine_candidate_input'],
        original_spectra_exact=True,queries=9,USED_only=True,scope=scope,
        quality_read=False,C13_read=False,scoring_entries=0))
    for name,source in [('repeat',local_input),('cloud-nine',out/'original-cloud-nine-input.json')]:
        dest=out/(name+'.json')
        command=[sys.executable,str(runner),str(package),str(source),str(dest),
            '--device','cuda','--budget','4000.0','--site',str(work/'ice_site'),
            '--wheels',str(base/'input/casmi26-attributed-v4g-reproduction-assets/casmi26-iceberg/wheels'),'--any-rdkit']
        with (out/(name+'.log')).open('w',encoding='utf8') as stream:
            result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=240)
        assert result.returncode==0
        meta=read(str(dest)+'.meta.json')
        assert meta['status']=='ok' and meta['n_mols_scored']==meta['n_mols_covered']==9
        assert meta['n_pred_fail']==meta['n_feat_fail']==0
        print('C75 GL ENGINEERING REPLAY DONE',name,meta['seconds'],flush=True)
    # Torch before Arrow, after both model subprocesses have exited.
    import torch
    import numpy as np
    import pandas as pd
    original=read(local_output);repeat=read(out/'repeat.json');subset=read(out/'cloud-nine.json')
    assert set(original)==set(repeat)==set(subset)==ids
    def comparison(a,b):
        pairs=[(a[mid][s],b[mid][s]) for mid in ids for s in a[mid].keys()&b[mid].keys()]
        missing=sum((u is None)!=(v is None) for u,v in pairs)
        values=[(u,v) for u,v in pairs if u is not None and v is not None]
        u,v=np.asarray(values,float).T
        return dict(common_finite_scores=len(u),missing_pattern_changes=missing,
            maximum_absolute_difference=float(np.abs(u-v).max(initial=0)),
            outside_1e5_tolerance=int((~np.isclose(u,v,rtol=1e-5,atol=1e-5)).sum()))
    cloud=pd.read_parquet(ROOT/'outputs/forks/c57-full-masked-validation-v1/c41_neural_p01.parquet')
    expected={mid:{r.smiles:float(r.gl) for r in group.itertuples(index=False) if np.isfinite(r.gl)}
              for mid,group in cloud[cloud.molecule_id.isin(ids)].groupby('molecule_id')}
    results=dict(same_input_repeat=comparison(original,repeat),
        original_cloud_candidate_input_vs_T4=comparison(subset,expected),
        local_candidate_input_vs_T4=comparison(original,expected))
    write(out/'results.json',dict(completed_utc=stamp(),source_sha256=sha(__file__),
        intent_sha256=sha(out/'intent.json'),results=results,
        files_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()},
        fitted_or_quality_selected=False,USED_only=True,new_source_quality_read=False,
        C13_read=False,scoring_entries=0))
    print('C75 GL INPUT/RUNTIME DIAGNOSIS',results,flush=True)


if __name__=='__main__':main()
