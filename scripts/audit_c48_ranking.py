"""Reconstruct every C48 permutation from scores, independently of serving code."""
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(name,'2')
import argparse,json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from c45_complete_parent import ROOT,read,write,sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('scores',type=Path);parser.add_argument('parent',type=Path);a=parser.parse_args()
    folder=a.scores;target=folder/'independent-ranking-audit.json';assert not target.exists()
    receipt=read(folder/'score-receipt.json');head=ROOT/'outputs/publish/c48-clean-residual-heads-20261112';manifest=read(head/'head-manifest.json')
    assert receipt['head_manifest_sha256']==sha(head/'head-manifest.json') and receipt['source_sha256']==manifest['files']['score_c48_crossmodal.py']
    assert receipt['baseline_sha256']==sha(a.parent)
    for name,pin in receipt['files_sha256'].items():assert sha(folder/name)==pin
    parent=pd.read_csv(a.parent,dtype=str,keep_default_na=False);assert len(parent)==receipt['queries']
    identity=pd.read_parquet(folder/'candidate-identities.parquet').set_index('smiles');raw=pd.read_parquet(folder/'candidate-scores.parquet')
    support=pd.read_csv(folder/'query-coverage.csv').set_index('molecule_id').supported.to_dict();reports={}
    for trial,entry in receipt['trials'].items():
        prefix=1 if trial=='cpu' else 2;assert entry['protected_parent_prefix']==prefix
        result=pd.read_csv(folder/(trial+'.csv'),dtype=str,keep_default_na=False);assert result.molecule_id.tolist()==parent.molecule_id.tolist()
        groups={mid:g.sort_values('parent_rank') for mid,g in raw[raw.trial.eq(trial)].groupby('molecule_id')};changed=fallback=0
        for old,new in zip(parent.itertuples(index=False),result.itertuples(index=False)):
            strings=old.smiles.split(';');values=groups[old.molecule_id]
            assert values.smiles.tolist()==strings and values.parent_rank.tolist()==list(range(1,len(strings)+1))
            formula=identity.loc[strings,'formula'].tolist();scores=values.score.to_numpy(float);eligible=values.eligible.to_numpy(bool);assert np.isfinite(scores).all()
            expected=list(range(len(strings)))
            if support[old.molecule_id]:
                for f in set(formula[prefix:]):
                    positions=np.asarray([i for i in range(prefix,len(strings)) if formula[i]==f])
                    if len(positions)<2 or not eligible[positions].all():continue
                    preference=positions[np.lexsort((positions,-scores[positions]))];rank={int(pos):j for j,pos in enumerate(preference,1)}
                    utility=np.asarray([1/(3+pos+1)+.2/(3+rank[int(pos)]) for pos in positions]);chosen=positions[np.lexsort((positions,-utility))]
                    for position,choice in zip(positions,chosen):expected[int(position)]=int(choice)
            else:fallback+=1
            actual=new.smiles.split(';');assert actual==[strings[i] for i in expected]
            assert actual[:prefix]==strings[:prefix] and set(actual)==set(strings)
            assert identity.loc[actual,'formula'].tolist()==formula
            keys=identity.loc[actual,'key'].tolist();assert 1<=len(keys)<=25 and len(set(keys))==len(keys)
            changed+=actual!=strings
        assert changed==entry['changed_lists']
        reports[trial]=dict(queries=len(parent),changed_lists=int(changed),protected_parent_prefix=prefix,
            independent_RRF_exact=True,unsupported_exact_fallbacks=fallback,rank1_changes=0,candidate_membership_changes=0,formula_slot_changes=0,official_identity_duplicates=0)
    write(target,dict(completed_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),score_receipt_sha256=sha(folder/'score-receipt.json'),
        parent_sha256=sha(a.parent),head_manifest_sha256=sha(head/'head-manifest.json'),trials=reports,quality_labels_read=False))
    print('C48 INDEPENDENT RANKING PASS',json.dumps(reports))


if __name__=='__main__':main()
