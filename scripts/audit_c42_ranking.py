"""Independently reconstruct C42 serving permutations from actual raw scores."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
import argparse,json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from precision_cache import ROOT,sha,write


def main():
    p=argparse.ArgumentParser();p.add_argument('scores',type=Path);p.add_argument('parent',type=Path);a=p.parse_args()
    out=a.scores;target=out/'independent-ranking-audit.json';assert not target.exists()
    receipt=json.loads((out/'score-receipt.json').read_text())
    for n,h in receipt['files_sha256'].items():assert sha(out/n)==h,n
    assert sha(a.parent)==receipt['baseline_sha256']
    parent=pd.read_csv(a.parent,dtype=str,keep_default_na=False)
    identities=pd.read_parquet(out/'candidate-identities.parquet').set_index('smiles')
    scores=pd.read_parquet(out/'candidate-scores.parquet')
    support=pd.read_csv(out/'query-coverage.csv').set_index('molecule_id').supported.to_dict()
    reports={}
    for trial in receipt['trials']:
        after=pd.read_csv(out/(trial+'.csv'),dtype=str,keep_default_na=False)
        assert after.molecule_id.tolist()==parent.molecule_id.tolist()
        groups={m:g.sort_values('parent_rank') for m,g in scores[scores.trial.eq(trial)].groupby('molecule_id')}
        changed=fallback=0
        for old,new in zip(parent.itertuples(index=False),after.itertuples(index=False)):
            ss=old.smiles.split(';');g=groups[old.molecule_id]
            assert g.smiles.tolist()==ss and g.parent_rank.tolist()==list(range(1,len(ss)+1))
            f=identities.loc[ss,'formula'].tolist();elig=g.eligible.to_numpy(bool);raw=g.score.to_numpy(float)
            assert np.isfinite(raw).all()
            expected=list(range(len(ss)))
            if support[old.molecule_id]:
                # Direct independent RRF reconstruction, using 1-based global
                # positions and head ranks inside eligible formula tail groups.
                for formula in set(f[1:]):
                    positions=np.asarray([i for i in range(1,len(ss)) if f[i]==formula])
                    if len(positions)<2 or not elig[positions].all():continue
                    preference=positions[np.lexsort((positions,-raw[positions]))]
                    head_rank={int(i):rank for rank,i in enumerate(preference,1)}
                    utilities=np.asarray([1/(3+(i+1))+.2/(3+head_rank[int(i)]) for i in positions])
                    chosen=positions[np.lexsort((positions,-utilities))]
                    for position,choice in zip(positions,chosen):expected[int(position)]=int(choice)
            else:fallback+=1
            actual=new.smiles.split(';');assert actual==[ss[i] for i in expected]
            assert actual[0]==ss[0] and set(actual)==set(ss)
            assert [identities.loc[s,'formula'] for s in actual]==f
            keys=identities.loc[actual,'key'].tolist();assert len(set(keys))==len(keys)
            changed+=actual!=ss
        assert changed==receipt['trials'][trial]['changed_lists']
        reports[trial]=dict(queries=len(parent),changed_lists=int(changed),unsupported_query_fallbacks=fallback,
            independent_RRF_exact=True,rank1_changes=0,candidate_membership_changes=0,formula_slot_changes=0,official_identity_duplicates=0)
    result=dict(completed_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),
        score_receipt_sha256=sha(out/'score-receipt.json'),parent_sha256=sha(a.parent),trials=reports,quality_labels_read=False)
    write(target,result);print('C42 INDEPENDENT RANKING PASS',json.dumps(reports),flush=True)


if __name__=='__main__':main()
