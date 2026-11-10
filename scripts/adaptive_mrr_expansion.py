"""Optimize one outsider's insertion position using a frozen posterior."""
import numpy as np
import pandas as pd


def expand(parent,evidence,config):
    assert parent.columns.tolist()==['molecule_id','smiles'] and parent.molecule_id.is_unique
    groups={str(mid):g.sort_values('prior_rank',kind='stable').reset_index(drop=True) for mid,g in evidence.groupby('molecule_id',sort=False)}
    result=parent.copy();actions=[]
    for index,row in parent.iterrows():
        before=row.smiles.split(';');n=len(before);assert 1<=n<=25
        g=groups.get(str(row.molecule_id));reason='no_pool';gain=0.;position=0
        if g is not None:
            assert g.iloc[:n].smiles.tolist()==before and g.iloc[:n].is_parent.astype(bool).all()
            assert g.metric_key.is_unique and not g.iloc[n:].is_parent.astype(bool).any()
            p=g.probability.to_numpy(float);lib=g.library_max.to_numpy(float)
            if not np.isfinite(lib).all():reason='missing_library_passthrough'
            elif not np.isfinite(p).all():reason='invalid_posterior_passthrough'
            elif lib[0]>=config['protect_library_at']:reason='protected_library'
            else:
                assert g.library_max.nunique()==1 and (p>=0).all() and p.sum()<=1.00001
                if len(g)==n:reason='no_outsider'
                else:
                    best=n+int(np.argmax(p[n:]));old=float((p[:n]/np.arange(1,n+1)).sum());options=[]
                    for pos in range(2,min(n+1,25)+1):
                        after=(list(range(pos-1))+[best]+list(range(pos-1,n)))[:25]
                        value=float((p[after]/np.arange(1,len(after)+1)).sum())-old
                        options.append((value,pos,after))
                    gain,chosen,after=max(options,key=lambda item:(item[0],item[1]))
                    if gain<=config['minimum_expected_mrr_gain']:reason='nonpositive_expected_gain'
                    else:
                        values=g.smiles.iloc[after].tolist()
                        assert values[0]==before[0] and [s for s in values if s in set(before)]==before[:24]
                        assert len(values)==min(n+1,25) and len(set(values))==len(values)
                        result.at[index,'smiles']=';'.join(values);reason='adaptive_expected_gain_inserted';position=chosen
        actions.append(dict(molecule_id=str(row.molecule_id),reason=reason,position=position,estimated_mrr_gain=gain))
    return result,pd.DataFrame(actions)
