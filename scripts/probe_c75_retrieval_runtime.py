"""Replay original USED12 Engine2 in the recovered cloud numerical runtime.

Only a diagnostic output filename changes in copied parent code. Reference
masks, scientific functions, public fit arrays and fixed query rows stay exact.
"""
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS']:
    os.environ[name]='2'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch
torch.set_num_threads(2)
import importlib.metadata as metadata,pickle,runpy,shutil,sys
from pathlib import Path
from c75_port_utils import ROOT,read,write,sha,stamp


def main():
    base=ROOT/'outputs/local/c75-local-core-port-20261112';out=base/'retrieval-runtime-probe-v1'
    assert not out.exists();out.mkdir()
    assert metadata.version('numpy')=='2.0.2' and metadata.version('numba')=='0.60.0'
    assert metadata.version('scikit-learn')=='1.6.1' and torch.__version__=='2.10.0+cu128'
    work=Path('/home/imsa/.local/share/enveda-c75/fixture/working')
    assert Path('/kaggle/working').resolve()==work
    source=work/'eng';code=out/'eng';code.mkdir()
    before={p.name:sha(p) for p in source.glob('*.py')}
    for name in before:shutil.copyfile(source/name,code/name)
    engine=code/'casmi_engine.py';text=engine.read_text(encoding='utf8')
    old='/kaggle/working/c41_engine2_reference_mask_audit.json';assert text.count(old)==1
    engine.write_text(text.replace(old,str(out/'reference-mask-audit.json')),encoding='utf8')
    mask=base/'fixture-port/query-package/engine2-raw-key-exclusion.json'
    os.environ['CASMI_C41_EXCLUSION_FILE']=str(mask)
    os.cpu_count=lambda:4
    inputs=Path('/kaggle/input/competitions/enveda-CASMI26-molecule-id-mass-spectra')
    original=base/'original-cloud-intermediates/eng_lists.json'
    write(out/'intent.json',dict(started_utc=stamp(),source_sha256=sha(__file__),original_engine_sources_sha256=before,
        copied_engine_sources_sha256={n:sha(code/n) for n in before},
        only_parent_code_change='Reference audit filename only; preserves the already completed parent run',
        test_sha256=sha(inputs/'test.parquet'),sample_sha256=sha(inputs/'sample_submission.csv'),mask_sha256=sha(mask),
        original_T4_lists_sha256=sha(original),original_local_lists_sha256=sha(work/'eng_lists.json'),
        versions={n:metadata.version(n) for n in ['torch','numpy','scipy','scikit-learn','numba']},
        original_parent_public_fit_arrays_only=True,USED_only=True,quality_read=False,C13_read=False,scoring_entries=0))
    sys.path.insert(0,str(code));import casmi_engine as E
    original_main=E.main
    def capture(*args,**kwargs):
        result=original_main(*args,**kwargs)
        with (out/'original-engine-return.pkl').open('wb') as stream:pickle.dump(result[1:],stream,protocol=5)
        return result
    E.main=capture
    sys.argv=[str(code/'eng_runner.py'),str(inputs/'test.parquet'),str(inputs/'train.parquet'),str(inputs/'sample_submission.csv'),str(out/'eng_lists.json')]
    runpy.run_path(str(code/'eng_runner.py'),run_name='__main__')
    actual=read(out/'eng_lists.json');cloud=read(original);previous=read(work/'eng_lists.json')
    ids=set(previous);assert set(actual)==ids and len(ids)==12
    comparisons={}
    for name,reference in [('original_T4',cloud),('original_local_runtime',previous)]:
        comparisons[name]=dict(queries=len(ids),exact_lists=sum(actual[q]==reference[q] for q in ids),
            exact_key_lists=sum(actual[q]['keys']==reference[q]['keys'] for q in ids),
            identical_key_sets=sum(set(actual[q]['keys'])==set(reference[q]['keys']) for q in ids),
            identical_leaders=sum(actual[q]['keys'][0]==reference[q]['keys'][0] for q in ids),
            changed_key_slots=sum(sum(a!=z for a,z in zip(actual[q]['keys'],reference[q]['keys'])) for q in ids))
    assert all(sha(source/n)==s for n,s in before.items())
    audit=read(out/'reference-mask-audit.json');assert audit['forbidden_rows_retained']==0
    write(out/'results.json',dict(completed_utc=stamp(),source_sha256=sha(__file__),intent_sha256=sha(out/'intent.json'),
        output_sha256=sha(out/'eng_lists.json'),comparison=comparisons,parent_source_files_unchanged=True,
        original_reference_mask_passed=True,original_engine_return_sha256=sha(out/'original-engine-return.pkl'),
        USED_only=True,quality_read=False,C13_read=False,scoring_entries=0,qualified_for_scoring=False))
    print('C75 RETRIEVAL RUNTIME COMPARISON',comparisons,flush=True)


if __name__=='__main__':main()
