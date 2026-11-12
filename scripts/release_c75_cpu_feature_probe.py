"""Freeze four public USED jobs and launch one CPU-only exact-image probe."""
import hashlib,json
from c75_port_utils import ROOT,read,write,sha,stamp


def main():
    a=read(ROOT/'.local/c38-active-goal.json');assert a['active'] and a['submissions_used']==0
    base=ROOT/'outputs/local/c75-local-core-port-20261112'
    out=base/'cpu-feature-parity';out.mkdir(exist_ok=True);assert not (out/'push-intent.json').exists()
    folder=ROOT/'kaggle/forks/c75-cpu-feature-parity'
    inp=base/'gl-engineering-probe-v1/original-cloud-nine-input.json';items=read(inp)
    candidates=sorted({s for item in items for s in item['cands']},key=lambda s:hashlib.sha256(('C75CPUfeature-v1:'+s).encode()).hexdigest())[:4]
    jobs=[dict(smiles=s,adduct='[M+H]+',instrument='QTOF',ce=40.) for s in candidates]
    source=folder/'feature_probe.py';text=source.read_text(encoding='utf8');marker='JOBS = []  # frozen_public_development_jobs'
    assert text.count(marker)==1;text=text.replace(marker,'JOBS = '+repr(jobs)+'  # frozen_public_development_jobs')
    compile(text,str(source),'exec');source.write_text(text,encoding='utf8',newline='\n')
    image=read(base/'cloud-runtime-metadata/actual-status.json')['actual_docker_image']
    meta=dict(id='clarkkitchen/casmi26-c75-cpu-feature-parity',title='casmi26 c75 cpu feature parity',code_file='feature_probe.py',
        language='python',kernel_type='script',is_private=True,enable_gpu=False,enable_tpu=False,enable_internet=False,
        dataset_sources=['dmitriigluzdov/casmi26-attributed-v4g-reproduction-assets'],competition_sources=[],kernel_sources=[],
        docker_image=image,docker_image_pinning_type='original')
    write(folder/'kernel-metadata.json',meta)
    write(out/'push-intent.json',dict(created_utc=stamp(),source_sha256=sha(__file__),kernel=meta['id'],expected_version=1,
        probe_source_sha256=sha(source),metadata_sha256=sha(folder/'kernel-metadata.json'),original_input_sha256=sha(inp),
        jobs=4,selection='First four by fixed hash of public USED candidate SMILES, without scores or truth',
        CPU_only=True,no_query_labels_or_competition_attached=True,scoring_entries=0))
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate();response=api.kernels_push(str(folder),timeout='900')
    write(out/'push-response.json',dict(received_utc=stamp(),response=json.loads(response.to_json())))
    result=dict(version=getattr(response,'version_number',None),error=getattr(response,'error',None),url=getattr(response,'url',None))
    write(out/'push-result.json',result);assert result['version']==1 and not result['error'],result
    print('C75 FOUR-JOB CPU FEATURE PARITY PROBE ACCEPTED',result,flush=True)


if __name__=='__main__':main()
