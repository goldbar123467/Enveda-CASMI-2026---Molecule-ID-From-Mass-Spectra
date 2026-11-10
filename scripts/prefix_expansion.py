"""Add bounded outsiders while retaining rank1 and the parent's original prefix."""
import numpy as np
import pandas as pd


def expand(parent, evidence, config):
    assert parent.columns.tolist() == ['molecule_id', 'smiles'] and parent.molecule_id.is_unique
    groups = {str(mid): group.sort_values('prior_rank', kind='stable').reset_index(drop=True)
        for mid, group in evidence.groupby('molecule_id', sort=False)}
    result = parent.copy(); actions = []
    for index, row in parent.iterrows():
        mid = str(row.molecule_id); before = row.smiles.split(';'); group = groups.get(mid)
        reason, count = 'no_pool', 0
        if group is not None:
            n = len(before)
            assert group.iloc[:n].smiles.tolist() == before and group.iloc[:n].is_parent.astype(bool).all()
            assert not group.iloc[n:].is_parent.astype(bool).any() and group.metric_key.is_unique
            if not np.isfinite(group.library_max.to_numpy(dtype=float)).all(): reason = 'missing_library_passthrough'
            elif not group.neural_ok.astype(bool).all(): reason = 'unsupported_model_passthrough'
            elif group.library_max.iloc[0] >= config['protect_library_at']: reason = 'protected_library'
            else:
                assert group.library_max.nunique() == 1
                scores = group.neural_score.to_numpy(dtype=float)
                assert np.isfinite(scores).all()
                order = np.argsort(-scores, kind='stable'); neural = np.empty(len(group), dtype=int)
                neural[order] = np.arange(1, len(group) + 1)
                fused = 1 / (config['rrf_offset'] + group.prior_rank.to_numpy()) + config['rrf_weight'] / (config['rrf_offset'] + neural)
                ranked = np.r_[0, 1 + np.argsort(-fused[1:], kind='stable')]
                slots = ranked[:min(25, len(ranked))]
                selected = set(int(i) for i in slots if i >= n)
                selected = set([int(i) for i in slots if int(i) in selected][:config['maximum_insertions']])
                count = len(selected)
                if not count: reason = 'no_outsider_selected'
                else:
                    kept = before[:min(n, 25 - count)]; cursor = 0; after = []
                    for i in slots:
                        if int(i) in selected: after.append(group.smiles.iloc[int(i)])
                        elif cursor < len(kept): after.append(kept[cursor]); cursor += 1
                    assert cursor == len(kept) and after[0] == before[0] and 1 <= len(after) <= 25
                    assert [s for s in after if s in set(before)] == kept
                    assert len(after) == len(set(after)) and len(after) == len(kept) + count
                    result.at[index, 'smiles'] = ';'.join(after); reason = 'inserted_outsiders'
        actions.append(dict(molecule_id=mid, reason=reason, inserted=count))
    return result, pd.DataFrame(actions)
