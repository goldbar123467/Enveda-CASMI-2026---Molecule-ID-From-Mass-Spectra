"""Apply the existing tail fusion only to explicitly negative query IDs."""
from compatible_tail import apply


def merge(parent,scores,modes,weight):
    assert parent.columns.tolist()==['molecule_id','smiles'] and parent.molecule_id.is_unique
    ids=set(parent.molecule_id.astype(str));assert ids==set(modes)
    assert set(modes.values())<= {'positive','negative'}
    negative={mid for mid in ids if modes[mid]=='negative'}
    assert set(scores.molecule_id.astype(str))==negative
    selected=parent[parent.molecule_id.astype(str).isin(negative)].copy()
    if negative:changed,report=apply(selected,scores,weight)
    else:
        changed=selected;report=dict(weight=weight,offset=3.,rank_one_changes=0,candidate_set_changes=0,
            changed_rows=0,scored_molecules=0,passthrough_molecules=0)
    result=parent.copy();lookup=changed.set_index('molecule_id').smiles.to_dict()
    for i,row in result.iterrows():
        if str(row.molecule_id) in negative:result.at[i,'smiles']=lookup[row.molecule_id]
    positive=~parent.molecule_id.astype(str).isin(negative)
    assert result.loc[positive].equals(parent.loc[positive])
    report.update(total_queries=len(parent),negative_queries=len(negative),positive_exact_passthrough=int(positive.sum()))
    return result,report
