"""Package verified C08 fits and an exact packed fingerprint cache; no upload."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import argparse,json,shutil
from precision_cache import ROOT,sha,write


def package(smoke=False):
    torch.set_num_threads(2)
    spec=json.loads((ROOT/'configs/compatible-training-20261108.json').read_text())
    run=ROOT/('outputs/local/smoke-compatible-20261108' if smoke else spec['output'])
    out=ROOT/('outputs/publish/c08-smoke-fixture' if smoke else 'outputs/publish/c08-models-20261108')
    steps=200 if smoke else 40000
    assert (run/'results.json').exists(),'All fits must finish before packaging'
    results=json.loads((run/'results.json').read_text());assert len(results['models'])==8
    manifest=json.loads((run/'manifest.json').read_text());assert manifest['smoke']==smoke
    for path,value in manifest['source_sha256'].items():assert sha(ROOT/path)==value,path
    out.mkdir(parents=True,exist_ok=True);models=[]
    for fold in [0,1]:
        for arm in spec['arms']:
            path=run/'models'/arm['name']/f'fold-{fold}'/'final_model.pt'
            receipt=json.loads(path.with_name('complete.json').read_text())
            assert receipt['steps']==steps and receipt['smoke']==smoke and receipt['heldout_in_training']==0
            assert receipt['checkpoint_sha256']==sha(path)
            ck=torch.load(path,map_location='cpu',weights_only=True)
            assert ck['steps']==steps and ck['smoke']==smoke and ck['fold']==fold and ck['cfg']['name']==arm['name']
            assert all(ck['cfg'][key]==value for key,value in arm.items());del ck
            name=f"{arm['name']}-fold{fold}.pt";shutil.copy2(path,out/name)
            models.append(dict(file=name,arm=arm['name'],fold=fold,cache_precision=arm['cache_precision'],steps=steps,
                training_identities=receipt['training_identities'],checkpoint_sha256=receipt['checkpoint_sha256']))
    for name in ['model.py','features.py']:shutil.copy2(ROOT/'outputs/local/kit'/name,out/name)
    for name in ['compatible_inference.py','compatible_tail.py']:shutil.copy2(ROOT/'scripts'/name,out/name)
    source=ROOT/spec['base_cache'];catalogue=pd.read_parquet(source/'molecules.parquet',columns=['smiles'])
    assert catalogue.smiles.is_unique,'Fingerprint string cache must be unambiguous'
    catalogue.to_parquet(out/'fingerprint-cache-smiles.parquet',index=False)
    original=np.load(source/'fingerprints.npy',mmap_mode='r');assert original.shape==(len(catalogue),2048)
    packed=np.lib.format.open_memmap(out/'fingerprint-cache.npy',mode='w+',dtype=np.uint8,shape=(len(catalogue),256))
    for start in range(0,len(original),4096):
        block=np.asarray(original[start:start+4096]);assert np.isin(block,[0,1]).all()
        value=np.packbits(block,axis=1,bitorder='big');assert np.array_equal(np.unpackbits(value,axis=1,bitorder='big'),block)
        packed[start:start+len(block)]=value
    packed.flush();del packed
    architecture={k:spec['training'][k] for k in ['width','heads','layers','embedding','fp_bits','max_peaks','min_peaks','temperature','fp_score_weight']}
    names=['model.py','features.py','compatible_inference.py','compatible_tail.py','fingerprint-cache-smiles.parquet','fingerprint-cache.npy']+[m['file'] for m in models]
    result=dict(kind='compatible_scorer',run_id='C08S' if smoke else 'C08',smoke=smoke,steps=steps,architecture=architecture,models=models,
        files={name:sha(out/name) for name in names},training_plan=spec,training_source_sha256=manifest['source_sha256'],
        original_fingerprint_cache_sha256=sha(source/'fingerprints.npy'),fingerprint_compression='np.packbits big, every bit round-trip verified',
        integration='C08 tail-only RRF after complete P01; rank1/candidate set protected',hidden_score_unmeasured=True)
    write(out/'c08-manifest.json',result)
    write(out/'dataset-metadata.json',dict(id='clarkkitchen/casmi26-c08-smoke-fixture' if smoke else 'clarkkitchen/casmi26-c08-models-20261108',
        title='CASMI26 C08 smoke fixture' if smoke else 'CASMI26 C08 models 20261108',licenses=[dict(name='CC-BY-NC-4.0')]))
    shutil.copy2(ROOT/'NOTICE.md',out/'NOTICE.md')
    (out/'README.md').write_text('# Private C08 compatible scorers\n\nEight independent fits,4 recipes x2 common folds. No averaged weights.\n'
        'Saner Turhaner architecture/features; Deotte common-fold scaffolding; bgunshui V7 architecture/negative-recipe inspiration. See NOTICE.md.\n'
        'Exact bit-packed fingerprints are an inference acceleration cache of supplied competition structures, not external labels.\n'
        f'Smoke fixture: {smoke}. Successful updates per fit: {steps}. Hidden improvement unproven.\n',encoding='utf-8')
    print('DONE C08 MODEL PACKAGE',out,sha(out/'c08-manifest.json'),flush=True);return out


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');package(parser.parse_args().smoke)
