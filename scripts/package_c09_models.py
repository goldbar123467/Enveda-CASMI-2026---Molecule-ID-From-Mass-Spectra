"""Package completed C09 continuations, retaining immutable C08 serving code."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
import torch
import json,hashlib,shutil
from pathlib import Path
from precision_cache import ROOT,sha,write


def scoring_source():
    source=(ROOT/'scripts/compatible_inference.py').read_text(encoding='utf-8')
    before='expected_steps=200 if smoke else 40000'
    assert source.count(before)==1
    return source.replace(before,"expected_steps=45000\n    assert not smoke and manifest['run_id']=='C09', 'Only completed C09 continuations are supported'")


def main():
    torch.set_num_threads(2)
    run=ROOT/'outputs/local/c09-20261109';out=ROOT/'outputs/publish/c09-models-20261109'
    results=json.loads((run/'results.json').read_text());assert len(results['models'])==2
    contract=json.loads((run/'source-contract.json').read_text())
    assert all(sha(ROOT/p)==digest for p,digest in contract['source_sha256'].items())
    parent=ROOT/'outputs/publish/c08-models-20261108';previous=json.loads((parent/'c08-manifest.json').read_text())
    out.mkdir(parents=True,exist_ok=True);models=[]
    for fold in [0,1]:
        folder=run/'models/served-score'/f'fold-{fold}';record=json.loads((folder/'complete.json').read_text())
        assert record['continuation_steps']==5000 and record['heldout_in_training']==0 and record['amp_skips']==0
        path=folder/'final_model.pt';assert sha(path)==record['checkpoint_sha256']
        value=torch.load(path,map_location='cpu',weights_only=True)
        assert value['steps']==45000 and value['continuation_steps']==5000 and value['fold']==fold and not value['smoke']
        assert value['parent_sha256']==record['parent_sha256']
        assert all(torch.isfinite(v).all() for v in value['state_dict'].values())
        assert value['cfg']['name']=='served-score' and value['cfg']['cache_precision']=='float16'
        entry=next(m for m in previous['models'] if m['arm']=='legacy16-hard' and m['fold']==fold)
        assert entry['checkpoint_sha256']==record['parent_sha256']
        name=f'served-score-fold{fold}.pt';shutil.copy2(path,out/name)
        models.append(dict(file=name,arm='served-score',fold=fold,cache_precision='float16',steps=45000,continuation_steps=5000,
            training_identities=record['training_identities'],checkpoint_sha256=sha(path),parent_sha256=record['parent_sha256']))
        del value
    for name in ['model.py','features.py','fingerprint-cache.npy','fingerprint-cache-smiles.parquet','compatible_tail.py']:
        assert sha(parent/name)==previous['files'][name],name
        shutil.copy2(parent/name,out/name)
    (out/'c09_inference.py').write_text(scoring_source(),encoding='utf-8')
    names=['model.py','features.py','fingerprint-cache.npy','fingerprint-cache-smiles.parquet','compatible_tail.py','c09_inference.py']+[m['file'] for m in models]
    manifest=dict(kind='compatible_scorer',run_id='C09',smoke=False,steps=45000,architecture=previous['architecture'],models=models,
        files={n:sha(out/n) for n in names},training_source_sha256=contract['source_sha256'],training_plan=contract['spec'],
        parent_manifest_sha256=sha(parent/'c08-manifest.json'),original_fingerprint_cache_sha256=previous['original_fingerprint_cache_sha256'],
        fingerprint_compression=previous['fingerprint_compression'],
        integration='Same complete-list scoring/arithmetic as C08; additional served-score-objective continuation; development-only tail weight; protect rank1/candidate membership',
        hidden_score_unmeasured=True,causal_limitation='Compared with initial C08 weights; extra updates and changed objective are not causally separated')
    write(out/'c08-manifest.json',manifest)
    write(out/'dataset-metadata.json',dict(id='clarkkitchen/casmi26-c09-models-20261109',title='CASMI26 C09 models 20261109',licenses=[dict(name='CC-BY-NC-4.0')]))
    write(run/'packaging.json',dict(completed_models=2,additional_updates=10000,all_finite=True,manifest_sha256=sha(out/'c08-manifest.json'),
        serving_derivation='One explicit expected-step/run-id guard change to pinned C08 complete-list scorer',source_sha256=sha(out/'c09_inference.py')))
    print('DONE C09 PACKAGING',json.dumps(json.loads((run/'packaging.json').read_text())),flush=True)


if __name__=='__main__':main()
