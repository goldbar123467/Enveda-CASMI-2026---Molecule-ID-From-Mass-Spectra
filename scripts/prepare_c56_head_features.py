"""Reuse the verified clean input boundary and refit only semantic targets."""
import os
for n in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(n,'2')
import argparse,shutil
import numpy as np
from c45_complete_parent import ROOT,read,write,sha,stamp
OUT=ROOT/'outputs/local/c56-bounded-semantic-heads-20261112'
OLD=ROOT/'outputs/local/c54-isomer-fingerprint-heads-20261112'


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);a=p.parse_args();config=ROOT/a.plan
    assert not (OUT/'head-feature-freeze.json').exists()
    assert read(ROOT/'outputs/local/c56-bounded-semantic-proxy-20261112/used-development-rule-results.json')['passed_pair']
    original=read(OLD/'head-feature-freeze.json');boundary=read(OLD/'fit-boundary-freeze.json')
    for name,pin in {**original['files_sha256'],**boundary['files_sha256']}.items():assert sha(OLD/name)==pin
    assert read(OLD/'independent-fitted-model-audit.json')['C39_C51_C53_and_global_C13_inventory_fit_identity_overlaps']==0
    assert sha(OUT/'fit-boundary-freeze.json')==sha(OLD/'fit-boundary-freeze.json')
    source=sha(__file__);cfg=sha(config);ids=np.load(OLD/'fitting-identities.npy');assert len(ids)==153401
    write(OUT/'head-feature-protocol.json',dict(frozen_utc=stamp(),source_sha256=source,config_sha256=cfg,
        exact_input_feature_source_sha256=sha(OLD/'head-feature-freeze.json'),exact_fit_boundary_sha256=sha(OLD/'fit-boundary-freeze.json'),
        no_old_fitted_head_weights_loaded=True,query_rows_used_for_fit=False,C53_quality_read=False,C13_read=False))
    shutil.copy2(OLD/'fit-features.npy',OUT/'fit-features.npy')
    raw=np.load(ROOT/'outputs/local/long-frozen-20261107-r1/molecule_embeddings.npy',mmap_mode='r')
    center=np.zeros(768,np.float64)
    for start in range(0,len(ids),8192):center+=np.asarray(raw[ids[start:start+8192]],np.float64).sum(0)
    center/=len(ids);scale=np.load(OLD/'fit-scaling.npz')
    np.savez(OUT/'fit-scaling.npz',input_mean=scale['input_mean'],input_std=scale['input_std'],molecule_center=center.astype(np.float32))
    targets=np.lib.format.open_memmap(OUT/'fit-targets.npy',mode='w+',dtype=np.float32,shape=(len(ids),768))
    for start in range(0,len(ids),4096):
        values=np.asarray(raw[ids[start:start+4096]],np.float64)-center;values/=np.linalg.norm(values,axis=1,keepdims=True);targets[start:start+4096]=values
    targets.flush();del targets
    assert sha(__file__)==source and sha(config)==cfg and sha(OUT/'fit-features.npy')==sha(OLD/'fit-features.npy')
    write(OUT/'head-feature-freeze.json',dict(completed_utc=stamp(),protocol_sha256=sha(OUT/'head-feature-protocol.json'),
        fitting_identities=len(ids),acquisitions=boundary['new_fit_spectrum_rows'],input_dimension=1039,target_dimension=768,
        files_sha256={name:sha(OUT/name) for name in ['fit-features.npy','fit-targets.npy','fit-scaling.npz']},
        stats_fitted_on_training_only=True,equal_identity_weights=True,C53_quality_read=False,C13_read=False,full_pipeline_validation_still_required=True))
    print('C56 CLEAN SEMANTIC FEATURES PREPARED',len(ids),flush=True)


if __name__=='__main__':main()
