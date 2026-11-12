"""Release one frozen CPU-only private sort probe; never score or retry."""
from c75_port_utils import ROOT,read,write,sha,stamp


def main():
    auth=read(ROOT/'.local/c38-active-goal.json');assert auth['active'] and auth['submissions_used']==0
    base=ROOT/'outputs/local/c75-local-core-port-20261112/cpu-sort-parity'
    pin=read(base/'freeze.json');folder=ROOT/'outputs/publish/c75-cpu-sort-parity-20261112'
    assert sha(folder/'sort_probe.py')==pin['probe_source_sha256'] and sha(folder/'kernel-metadata.json')==pin['metadata_sha256']
    meta=read(folder/'kernel-metadata.json');assert not meta['enable_gpu'] and not meta['dataset_sources'] and not meta['competition_sources']
    intent=base/'push-intent.json';assert not intent.exists(),'Reconcile prior intent instead of duplicate push'
    write(intent,dict(created_utc=stamp(),source_sha256=sha(__file__),kernel=meta['id'],version=1,freeze_sha256=sha(base/'freeze.json'),scoring_entry=False))
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate();response=api.kernels_push(str(folder),timeout='600')
    write(base/'push-response.json',dict(received_utc=stamp(),response=__import__('json').loads(response.to_json())))
    result=dict(version=getattr(response,'version_number',None),error=getattr(response,'error',None),url=getattr(response,'url',None))
    write(base/'push-result.json',result);assert result['version']==1 and not result['error'],result
    print('C75 PRIVATE CPU SORT PROBE ACCEPTED',result,flush=True)


if __name__=='__main__':main()
