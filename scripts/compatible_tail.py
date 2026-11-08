"""Conservative C08 tail-only reciprocal-rank fusion after the complete P01 run."""
import numpy as np
import pandas as pd


def order(scores,weight,offset=3.):
    """Keep rank1 exactly; stable ties favor the existing P01 candidate order."""
    values=np.asarray(scores,np.float64);n=len(values)
    assert np.isfinite(values).all() and 0<=weight<=.2
    if n<=1 or weight==0:return np.arange(n)
    ranked=np.argsort(-values,kind='stable');neural=np.empty(n,np.int64);neural[ranked]=np.arange(1,n+1)
    fusion=1/(offset+np.arange(1,n+1))+weight/(offset+neural)
    return np.r_[0,1+np.argsort(-fusion[1:],kind='stable')]


def apply(baseline,scores,weight):
    assert baseline.columns.tolist()==['molecule_id','smiles'] and baseline.molecule_id.is_unique
    groups={str(mid):g.sort_values('baseline_rank',kind='stable') for mid,g in scores.groupby('molecule_id',sort=False)}
    rows=[];changed=used=fallback=0
    for row in baseline.itertuples(index=False):
        mid=str(row.molecule_id);before=row.smiles.split(';');group=groups[mid]
        assert group.smiles.tolist()==before and not group.smiles.duplicated().any()
        if group.c08_ok.astype(bool).all():
            indices=order(group.c08_score.to_numpy(),weight);after=[before[i] for i in indices];used+=1
        else:after=before;fallback+=1
        assert after[0]==before[0] and len(after)==len(before) and set(after)==set(before)
        changed+=int(after!=before);rows.append(dict(molecule_id=mid,smiles=';'.join(after)))
    return pd.DataFrame(rows),dict(weight=weight,offset=3.,rank_one_changes=0,candidate_set_changes=0,
        changed_rows=changed,scored_molecules=used,passthrough_molecules=fallback)
