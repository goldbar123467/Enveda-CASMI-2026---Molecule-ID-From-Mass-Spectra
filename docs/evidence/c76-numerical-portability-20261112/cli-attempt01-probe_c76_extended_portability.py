"""Compare C76 CPU/CUDA on all799scores of the earlier frozen USED-nine input."""
import os,subprocess,sys
from c75_port_utils import ROOT,read,write,sha,stamp


def main():
    base=ROOT/'outputs/local/c75-local-core-port-20261112';out=ROOT/'outputs/local/c76-numerical-portability-20261112/extended-used-nine'
    assert not out.exists();out.mkdir(parents=True)
    pin=read(out.parent/'candidate-freeze.json')
    for name,value in pin['files_sha256'].items():assert sha(ROOT/name)==value,name
    source=base/'gl-engineering-probe-v1/original-cloud-nine-input.json'
    assert sha(source)==read(base/'gl-engineering-probe-v1/intent.json')['cloud_nine_input_sha256']
    pkg=base/'input/casmi26-attributed-v4g-reproduction-assets/casmi26-glacier'
    write(out/'intent.json',dict(started_utc=stamp(),source_sha256=sha(__file__),candidate_freeze_sha256=sha(out.parent/'candidate-freeze.json'),
        input_sha256=sha(source),original_frozen_queries=9,expected_scores=799,
        acceptance='Both executions healthy9of9; same799scorekeys/nullpatterns and atol=rtol=1e-5',
        original_C75_gate_still_failed=True,USED_only=True,quality_read=False,C13_read=False,scoring_entries=0))
    values={};metas={}
    for device in ['cpu','cuda']:
        output=out/(device+'.json')
        command=[sys.executable,str(ROOT/'scripts/run_c76_gl_portable.py'),str(pkg),str(source),str(output),
            '--device',device,'--budget','4000.0','--site','/home/imsa/.local/share/enveda-c75/fixture/working/ice_site',
            '--wheels',str(pkg.parent/'casmi26-iceberg/wheels'),'--any-rdkit']
        with (out/(device+'.log')).open('w',encoding='utf8') as stream:
            run=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=600)
        assert run.returncode==0,run.returncode
        meta=read(str(output)+'.meta.json');assert meta['status']=='ok' and meta['n_mols_scored']==meta['n_mols_covered']==9
        assert meta['n_pred_fail']==meta['n_feat_fail']==0 and meta['device']==device
        assert read(str(output)+'.portability.json')['weights_exact']
        values[device]=read(output);metas[device]=meta
    a=values['cpu'];z=values['cuda'];assert a.keys()==z.keys()
    pairs=[];null_mismatch=0
    for mid,entries in a.items():
        assert entries.keys()==z[mid].keys()
        for candidate,v in entries.items():
            w=z[mid][candidate]
            if v is None or w is None:null_mismatch+=int(v is not w)
            else:pairs.append((v,w))
    assert len(pairs)==799 and null_mismatch==0
    comparison=dict(finite_scores=len(pairs),null_mismatches=null_mismatch,max_absolute_difference=max(abs(x-y) for x,y in pairs),
        outside_original_tolerance=sum(abs(x-y)>1e-5+1e-5*abs(y) for x,y in pairs))
    write(out/'results.json',dict(completed_utc=stamp(),source_sha256=sha(__file__),intent_sha256=sha(out/'intent.json'),
        comparison=comparison,component_passed=comparison['outside_original_tolerance']==0,
        original_C75_gate_passed=False,qualified_for_scoring=False,
        files_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()},USED_only=True,quality_read=False,C13_read=False,scoring_entries=0))
    print('C76 EXTENDED USED COMPONENT COMPARISON',comparison,flush=True)


if __name__=='__main__':main()
