"""One-source, joint-top-choice outsider insertion with exact parent fallback."""
import numpy as np
import pandas as pd


def expand(parent,evidence,config):
    assert parent.columns.tolist() == ['molecule_id','smiles'] and parent.molecule_id.is_unique
    groups = {str(mid):g.sort_values('prior_rank',kind='stable').reset_index(drop=True)
        for mid,g in evidence.groupby('molecule_id',sort=False)}
    result = parent.copy(); actions = []
    for index,row in parent.iterrows():
        before = row.smiles.split(';'); group = groups.get(str(row.molecule_id)); reason = 'no_pool'; inserted = 0
        if group is not None:
            n = len(before)
            assert group.iloc[:n].smiles.tolist() == before and group.iloc[:n].is_parent.astype(bool).all()
            assert not group.iloc[n:].is_parent.astype(bool).any() and group.metric_key.is_unique
            library = group.library_max.to_numpy(dtype=float)
            scores = group[['public_score','v80_score']].to_numpy(dtype=float)
            if not np.isfinite(library).all(): reason = 'missing_library_passthrough'
            elif not group.neural_ok.astype(bool).all() or not np.isfinite(scores).all(): reason = 'unsupported_model_passthrough'
            elif library[0] >= config['protect_library_at']: reason = 'protected_library'
            else:
                assert group.library_max.nunique() == 1
                top = np.argmax(scores,axis=0)
                if top[0] != top[1]: reason = 'model_disagreement'
                elif int(top[0]) < n: reason = 'top_is_parent'
                elif not bool(group.is_lotus.iloc[int(top[0])]): reason = 'top_not_lotus'
                else:
                    margins = np.sort(scores,axis=0)[-1]-np.sort(scores,axis=0)[-2]
                    if not (margins > config['minimum_raw_top_margin']).all(): reason = 'ambiguous_top_passthrough'
                    else:
                        outsider = group.smiles.iloc[int(top[0])]
                        after = before[:1]+[outsider]+before[1:24]
                        assert after[0] == before[0] and [s for s in after if s in set(before)] == before[:24]
                        assert len(after) == min(n+1,25) and len(set(after)) == len(after)
                        result.at[index,'smiles'] = ';'.join(after); inserted = 1; reason = 'joint_top_lotus_inserted'
        actions.append(dict(molecule_id=str(row.molecule_id),reason=reason,inserted=inserted))
    return result,pd.DataFrame(actions)
