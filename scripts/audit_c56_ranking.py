"""Independently reconstruct frozen C49 serving orders, without quality labels."""
import os
for n in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(n,'2')
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from c45_complete_parent import ROOT,read,write,sha,stamp


def main():
    p=argparse.ArgumentParser();p.add_argument('scores',type=Path);p.add_argument('parent',type=Path);a=p.parse_args()
    target=a.scores/'independent-ranking-audit.json';assert not target.exists()
    bundle=ROOT/'outputs/publish/c56-bounded-semantic-heads-20261112';manifest=read(bundle/'head-manifest.json')
    receipt=read(a.scores/'score-receipt.json')
    assert receipt['head_manifest_sha256']==sha(bundle/'head-manifest.json')
    assert receipt['source_sha256']==manifest['files']['score_c56_bounded_semantic.py'] and receipt['baseline_sha256']==sha(a.parent)
    for name,pin in receipt['files_sha256'].items():assert sha(a.scores/name)==pin,name
    parent=pd.read_csv(a.parent,dtype=str,keep_default_na=False);assert len(parent)==receipt['queries']
    ids=pd.read_parquet(a.scores/'candidate-identities.parquet').set_index('smiles')
    raw=pd.read_parquet(a.scores/'candidate-scores.parquet')
    support=pd.read_csv(a.scores/'query-coverage.csv').set_index('molecule_id').supported.to_dict();reports={}
    for trial,entry in receipt['trials'].items():
        assert entry['protected_parent_prefix']==2 and entry['last_reorderable_parent_rank']==10 and entry['consensus_applied']==(trial=='gpu')
        answer=pd.read_csv(a.scores/(trial+'.csv'),dtype=str,keep_default_na=False)
        assert answer.molecule_id.tolist()==parent.molecule_id.tolist()
        groups={mid:g.sort_values('parent_rank') for mid,g in raw[raw.trial.eq(trial)].groupby('molecule_id')};changed=fallback=0
        for old,new in zip(parent.itertuples(index=False),answer.itertuples(index=False)):
            strings=old.smiles.split(';');table=groups[old.molecule_id]
            assert table.smiles.tolist()==strings and table.parent_rank.tolist()==list(range(1,len(strings)+1))
            formulas=ids.loc[strings,'formula'].tolist();eligible=table.eligible.to_numpy(bool)
            score=table.score.to_numpy(float);reference=table.reference_score.to_numpy(float)
            assert np.isfinite(score).all() and np.isfinite(reference).all();expected=list(range(len(strings)))
            if support[old.molecule_id]:
                for f in set(formulas[2:10]):
                    positions=[i for i in range(2,min(10,len(strings))) if formulas[i]==f]
                    if len(positions)<2 or not eligible[positions].all():continue
                    utilities=[]
                    for values in [score,reference] if trial=='gpu' else [score]:
                        preference=sorted(positions,key=lambda i:(-values[i],i));ranks={i:j for j,i in enumerate(preference,1)}
                        utilities.append({i:1/(3+i+1)+.2/(3+ranks[i]) for i in positions})
                    # Insertion normal form is independent of serving bubble passes.
                    chosen=[]
                    for i in positions:
                        chosen.append(i);j=len(chosen)-1
                        while j and all(u[chosen[j]]>u[chosen[j-1]] for u in utilities):
                            chosen[j],chosen[j-1]=chosen[j-1],chosen[j];j-=1
                    order={i:j for j,i in enumerate(chosen)}
                    for i in positions:
                        for j in positions:
                            if i<j and order[j]<order[i]:assert all(u[j]>u[i] for u in utilities)
                    for i,j in zip(positions,chosen):expected[i]=j
            else:fallback+=1
            actual=new.smiles.split(';');assert actual==[strings[i] for i in expected]
            assert actual[:2]==strings[:2] and actual[10:]==strings[10:] and set(actual)==set(strings) and ids.loc[actual,'formula'].tolist()==formulas
            keys=ids.loc[actual,'key'].tolist();assert 1<=len(keys)<=25 and len(set(keys))==len(keys)
            changed+=actual!=strings
        assert changed==entry['changed_lists']
        reports[trial]=dict(queries=len(parent),changed_lists=int(changed),protected_parent_prefix=2,last_reorderable_parent_rank=10,
            independent_insertion_order_exact=True,all_inversions_supported_by_both_heads=trial=='gpu',
            unsupported_exact_fallbacks=fallback,rank1_changes=0,candidate_membership_changes=0,formula_slot_changes=0,official_identity_duplicates=0)
    write(target,dict(completed_utc=stamp(),source_sha256=sha(__file__),score_receipt_sha256=sha(a.scores/'score-receipt.json'),
        parent_sha256=sha(a.parent),head_manifest_sha256=sha(bundle/'head-manifest.json'),trials=reports,quality_labels_read=False,C13_read=False))
    print('C56 INDEPENDENT RANKING PASS',reports,flush=True)


if __name__=='__main__':main()
