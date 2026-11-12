"""Bind unchanged outcome-blind feature parity to the strict clean OOF pair."""
import os
for n in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(n,'2')
import torch
import numpy as np
import pandas as pd
from c45_complete_parent import ROOT,read,write,sha,stamp
from c67_ranker import predict


def main():
    data=ROOT/'outputs/local/c69-clean-parent-evidence-oof-20261112';dev=ROOT/'outputs/local/c70-clean-gpu-parent-evidence-oof-20261112'
    previous=ROOT/'outputs/local/c67-gpu-parent-evidence-oof-20261112/runtime-parity-audit.json'
    prior=read(previous);assert prior['feature_values_bitwise_exact'] and prior['both_5fold_candidate_lists_exact']
    assert prior['runtime_sha256']==sha(ROOT/'scripts/c67_ranker.py')
    original=ROOT/'outputs/local/c66-parent-evidence-oof-20261112'
    for name in ['features.npy','feature-names.json']:assert sha(data/name)==sha(original/name)
    rows=pd.read_parquet(data/'candidates.parquet');q=pd.read_parquet(data/'queries.parquet');x=np.load(data/'features.npy')
    for name,frame in [('queries.parquet',q),('candidates.parquet',rows)]:
        old=pd.read_parquet(original/name);pd.testing.assert_frame_equal(frame[old.columns],old,check_exact=True)
    for tag,folder,gpu in [('cpu',data,False),('gpu',dev,True)]:
        actual=pd.read_csv(folder/f'{tag}-oof.csv',dtype=str).set_index('molecule_id')
        allscores=pd.read_parquet(folder/f'{tag}-oof-scores.parquet')
        for fold in range(5):
            take=rows.fold.eq(fold);r=rows[take].reset_index(drop=True)
            lists,scores=predict(r,x[take],np.load(folder/f'{tag}-fold{fold}.npz'),gpu)
            np.testing.assert_allclose(scores,allscores.loc[take,'oof_score'],rtol=0,atol=1e-6)
            assert all(';'.join(v)==actual.loc[mid,'smiles'] for mid,v in lists.items())
    write(dev/'runtime-parity-audit.json',dict(completed_utc=stamp(),source_sha256=sha(__file__),runtime_sha256=sha(ROOT/'scripts/c67_ranker.py'),
        full_original_acquisition_feature_parity_receipt_sha256=sha(previous),features=101,candidates=len(rows),queries=len(q),
        original_unlabeled_acquisitions_only=True,feature_values_bitwise_exact=True,both_5fold_candidate_lists_exact=True,
        reserved_clean_fitting_not_evaluation_filter=True,C13_read=False,scoring_entries=0))
    print('C70 CLEAN CPU/GPU RUNTIME PARITY ALL869/21719/101 PASS',flush=True)


if __name__=='__main__':main()
