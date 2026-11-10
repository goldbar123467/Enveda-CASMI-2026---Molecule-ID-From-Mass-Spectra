"""One rank2 insertion selected by posterior expected reciprocal-rank utility."""
import numpy as np
import pandas as pd


def expand(parent,evidence,config):
    assert parent.columns.tolist()==['molecule_id','smiles'] and parent.molecule_id.is_unique
    groups={str(mid):g.sort_values('prior_rank',kind='stable').reset_index(drop=True)
        for mid,g in evidence.groupby('molecule_id',sort=False)}
    result=parent.copy();actions=[]
    for index,row in parent.iterrows():
        before=row.smiles.split(';');n=len(before);assert 1<=n<=25
        g=groups.get(str(row.molecule_id));reason='no_pool';count=0;gain=0.
        if g is not None:
            assert g.iloc[:n].smiles.tolist()==before and g.iloc[:n].is_parent.astype(bool).all()
            assert g.metric_key.is_unique and not g.iloc[n:].is_parent.astype(bool).any()
            probability=g.probability.to_numpy(dtype=float);library=g.library_max.to_numpy(dtype=float)
            if not np.isfinite(library).all():reason='missing_library_passthrough'
            elif not np.isfinite(probability).all():reason='invalid_posterior_passthrough'
            elif library[0]>=config['protect_library_at']:reason='protected_library'
            else:
                assert g.library_max.nunique()==1 and (probability>=0).all() and probability.sum()<=1.00001
                if len(g)==n:reason='no_outsider'
                else:
                    best=n+int(np.argmax(probability[n:]))
                    after=[0,best]+list(range(1,min(n,24)))
                    old=float((probability[:n]/np.arange(1,n+1)).sum())
                    new=float((probability[after]/np.arange(1,len(after)+1)).sum());gain=new-old
                    if gain<=config['minimum_expected_mrr_gain']:reason='nonpositive_expected_gain'
                    else:
                        values=[g.smiles.iloc[i] for i in after]
                        assert values[0]==before[0] and [s for s in values if s in set(before)]==before[:24]
                        assert len(values)==min(n+1,25) and len(set(values))==len(values)
                        result.at[index,'smiles']=';'.join(values);reason='expected_gain_inserted';count=1
        actions.append(dict(molecule_id=str(row.molecule_id),reason=reason,inserted=count,estimated_mrr_gain=gain))
    return result,pd.DataFrame(actions)
