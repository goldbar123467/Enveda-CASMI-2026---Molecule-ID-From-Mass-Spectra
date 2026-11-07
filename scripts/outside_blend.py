"""Blend public/private outsider scores, including duplicate and failed rows."""
import numpy as np
import pandas as pd


def keyed(frame):
    frame=frame.copy()
    frame['molecule_id']=frame.molecule_id.astype(str)
    frame['_occurrence']=frame.groupby(['molecule_id','smiles'],sort=False).cumcount()
    return frame.set_index(['molecule_id','smiles','_occurrence'])


def z(values):
    groups=values.groupby(level=0)
    return (values-groups.transform('mean'))/groups.transform(lambda value:value.std(ddof=0)).clip(lower=1e-6)


def combine(public,private):
    a,b=keyed(public),keyed(private)
    assert len(a)==len(b) and set(a.index)==set(b.index),'Outsider candidate keys differ'
    b=b.reindex(a.index);result=a.copy()
    ok_a=a.neural_ok.fillna(False).astype(bool);ok_b=b.neural_ok.fillna(False).astype(bool)
    both=ok_a&ok_b
    za,zb=z(a.neural_score.fillna(0)),z(b.neural_score.fillna(0))
    result['neural_score']=np.where(both,.5*za+.5*zb,np.where(ok_a,za,np.where(ok_b,zb,0)))
    for column in ['neural_cos','neural_fp','neural_std']:
        result[column]=np.where(both,.5*a[column]+.5*b[column],np.where(ok_a,a[column],np.where(ok_b,b[column],0)))
    result['neural_ok']=ok_a|ok_b
    assert np.isfinite(result.neural_score).all()
    return result.reset_index().drop(columns='_occurrence')
