"""Freeze and release one exact-image CPU proof of the explicit C76 adapter."""
from c75_port_utils import ROOT,read,write,sha,stamp


def main():
    auth=read(ROOT/'.local/c38-active-goal.json');assert auth['active'] and auth['submissions_used']==0
    base=ROOT/'outputs/local/c76-numerical-portability-20261112';pin=read(base/'candidate-freeze-r1.json')
    helper=ROOT/'scripts/c76_gl_portability.py';assert sha(helper)==pin['files_sha256']['scripts/c76_gl_portability.py']
    old=ROOT/'kaggle/forks/c75-cpu-feature-parity/feature_probe.py'
    assert sha(old)==read(ROOT/'outputs/local/c75-local-core-port-20261112/cpu-feature-parity/push-intent.json')['probe_source_sha256']
    text=old.read_text(encoding='utf8')
    needle="model=g.load_model(str(pkg),'cpu',print);fz=g.Featurizer(model)";assert text.count(needle)==1
    replacement="model=g.load_model(str(pkg),'cpu',print);cpu=g.load_model(str(pkg),'cpu',print)\n    portable={};exec("+repr(helper.read_text(encoding='utf8'))+",portable)\n    adapter=portable['install'](model,cpu);fz=g.Featurizer(model)"
    text=text.replace(needle,replacement)
    needle='result=dict(python=platform.python_version(),source_sha256=sha(__file__),jobs=JOBS,canonical=canonical,';assert text.count(needle)==1
    text=text.replace(needle,needle+'\n        c76_candidate_freeze_sha256='+repr(sha(base/'candidate-freeze-r1.json'))+',c76_helper_sha256='+repr(sha(helper))+',c76_adapter=adapter,')
    compile(text,'c76_cpu_feature_probe','exec')
    folder=ROOT/'kaggle/forks/c76-cpu-feature-parity';assert not folder.exists();folder.mkdir()
    source=folder/'feature_probe.py';source.write_text(text,encoding='utf8',newline='\n')
    meta=read(ROOT/'kaggle/forks/c75-cpu-feature-parity/kernel-metadata.json')
    meta.update(id='clarkkitchen/casmi26-c76-cpu-feature-parity',title='casmi26 c76 cpu feature parity')
    write(folder/'kernel-metadata.json',meta)
    out=base/'cpu-feature-parity';assert not out.exists();out.mkdir()
    write(out/'freeze.json',dict(frozen_utc=stamp(),source_sha256=sha(__file__),probe_source_sha256=sha(source),metadata_sha256=sha(folder/'kernel-metadata.json'),
        c76_candidate_freeze_sha256=sha(base/'candidate-freeze-r1.json'),c76_helper_sha256=sha(helper),original_probe_source_sha256=sha(old),
        exact_original4USED_jobs=True,actual_CPU_only=True,quality_read=False,C13_read=False,scoring_entries=0))
    write(out/'push-intent.json',dict(created_utc=stamp(),source_sha256=sha(__file__),kernel=meta['id'],expected_version=1,
        probe_source_sha256=sha(source),metadata_sha256=sha(folder/'kernel-metadata.json'),freeze_sha256=sha(out/'freeze.json'),scoring_entries=0))
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate();response=api.kernels_push(str(folder),timeout='900')
    write(out/'push-response.json',dict(received_utc=stamp(),response=__import__('json').loads(response.to_json())))
    result=dict(version=getattr(response,'version_number',None),error=getattr(response,'error',None),url=getattr(response,'url',None))
    write(out/'push-result.json',result);assert result['version']==1 and not result['error'],result
    print('C76 PRIVATE EXACT-IMAGE CPU PORTABILITY PROBE ACCEPTED',result,flush=True)


if __name__=='__main__':main()
