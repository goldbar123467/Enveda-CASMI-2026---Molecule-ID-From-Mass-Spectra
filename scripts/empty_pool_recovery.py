"""Fingerprint existing Engine1 proposals only for genuinely empty Engine2 pools."""
import json
from pathlib import Path
import numpy as np
import pandas as pd


def augment(pool,queries,proposals,engine,audit_path):
    assert len(queries) and set(queries.molecule_id.astype(str))==set(proposals)
    assert float(engine.CFG.PPM_FALLBACK)==30.0
    original_size=len(pool['mass']);known=set(pool['keys']);added={};reports=[]
    for mid,g in queries.groupby('molecule_id',sort=True):
        mid=str(mid);neutral=engine.pv.neutral_mass(g.precursor_mz.to_numpy(float),g.adduct.astype(str).to_numpy())
        assert np.isfinite(neutral).all() and (neutral>0).all()
        target=float(np.median(neutral));width=target*30e-6
        existing=int(((pool['mass']>=target-width)&(pool['mass']<=target+width)).sum())
        assert existing==0,'Recovery is forbidden when the original Engine2 pool has candidates'
        candidates=proposals[mid];assert 1<=len(candidates)<=60
        counts=dict(proposed=len(candidates),invalid=0,mass_rejected=0,already_known=0,added=0)
        for smiles in candidates:
            value=engine.fp_and_mass(smiles);key=engine.canon_key(smiles)
            if value is None or not key:counts['invalid']+=1;continue
            fp,mass=value;mass=float(mass)
            assert np.asarray(fp).shape==(pool['fp'].shape[1],) and np.asarray(fp).dtype==np.uint8 and np.isfinite(mass)
            if abs(mass-target)>width:counts['mass_rejected']+=1;continue
            if key in known or key in added:counts['already_known']+=1;continue
            added[key]=(str(smiles),mass,np.asarray(fp));counts['added']+=1
        assert counts['added']>0,'No usable real Engine1 proposals; never insert placeholders or weaken coverage'
        reports.append(dict(molecule_id=mid,target=target,original_candidates=existing,**counts))
    keys=list(added);mass=np.r_[pool['mass'],[added[k][1] for k in keys]]
    order=np.argsort(mass,kind='stable');result=dict(pool)
    result['mass']=mass[order]
    result['keys']=np.concatenate([pool['keys'],np.asarray(keys,dtype=object)])[order]
    result['smiles']=np.concatenate([pool['smiles'],np.asarray([added[k][0] for k in keys],dtype=object)])[order]
    result['fp']=np.concatenate([pool['fp'],np.stack([added[k][2] for k in keys])],axis=0)[order]
    result['k2i']=pd.Series(np.arange(len(order)),index=result['keys']);result['k2i']=result['k2i'][~result['k2i'].index.duplicated()]
    for row in reports:
        count=int((np.abs(result['mass']-row['target'])<=row['target']*30e-6).sum());assert count>0;row['recovered_candidates']=count
    audit=dict(original_pool_size=original_size,added_structures=len(added),queries=reports,
        source='Existing Engine1 BASE top60 generated proposals; no query structure labels',
        original_engine2_pool_and_ranker_rules_retained=True,only_empty_pool_queries=True,
        original_30ppm_fallback_window_retained=True,query_rows_dropped=0,placeholders_added=0,
        prediction_label_metrics_read=False)
    Path(audit_path).write_text(json.dumps(audit,indent=2),encoding='utf-8')
    return result
