"""Input classification and exact parent preservation for the fixed public recipe."""
import numpy as np
import pandas as pd


def intake(parent,queries,features,known_adducts):
    assert parent.columns.tolist()==['molecule_id','smiles'] and parent.molecule_id.is_unique
    assert {'molecule_id','ionization_mode','adduct','precursor_mz','ms2_mzs','ms2_normalized_intensities'}<=set(queries)
    q=queries.copy();q.molecule_id=q.molecule_id.astype(str)
    groups={str(mid):g for mid,g in q.groupby('molecule_id',sort=False)}
    known=set(known_adducts)|set(features.ADDUCTS)
    eligible=set();rows=[]
    for mid in parent.molecule_id.astype(str):
        g=groups.get(mid);reason='supported';mode=None;usable=0
        if g is None:reason='absent'
        else:
            modes={str(x).strip().lower() for x in g.ionization_mode}
            if not modes<={'positive','negative'}:reason='unknown_mode'
            else:
                # The public all-mode recipe combines usable acquisitions from
                # both polarities. Mixed polarity is supported, unlike C26's
                # earlier negative-only adapter.
                mode=next(iter(modes)) if len(modes)==1 else 'mixed'
                # Recognition alone accepts boundary whitespace; the original
                # text still reaches features.metadata, retaining its encoding.
                if not g.adduct.astype(str).str.strip().isin(known).all():reason='unknown_adduct'
                else:
                    for row in g.itertuples(index=False):
                        try:
                            precursor=float(row.precursor_mz)
                            if not np.isfinite(precursor) or precursor<=0:reason='invalid_precursor';break
                            mz,it=features.array_value(row.ms2_mzs),features.array_value(row.ms2_normalized_intensities)
                            if len(mz)!=len(it):reason='invalid_spectrum';break
                            if int((np.isfinite(mz)&np.isfinite(it)&(mz>0)&(it>0)).sum())<3:continue
                            x,n=features.peaks(row.ms2_mzs,row.ms2_normalized_intensities,precursor,96)
                            meta,_=features.metadata(precursor,row.adduct,row.ionization_mode,getattr(row,'collision_energy_ev',np.nan))
                            if n<3:continue
                            if not np.isfinite(x).all() or not np.isfinite(meta).all():reason='nonfinite_features';break
                            usable+=1
                        except (ValueError,TypeError,OverflowError):reason='invalid_spectrum';break
                    if reason=='supported' and not usable:reason='too_few_peaks'
        if reason=='supported':eligible.add(mid)
        rows.append(dict(molecule_id=mid,mode=mode,reason=reason,eligible=reason=='supported',spectrum_rows=0 if g is None else len(g),usable_spectrum_rows=usable))
    return eligible,pd.DataFrame(rows)


def protect(parent,candidate,eligible):
    """Only supported IDs can change; protected strings remain byte-equivalent."""
    assert parent.columns.tolist()==candidate.columns.tolist()==['molecule_id','smiles']
    assert parent.molecule_id.is_unique and candidate.molecule_id.is_unique
    assert candidate.molecule_id.tolist()==parent.molecule_id.tolist()
    assert set(eligible)<=set(parent.molecule_id)
    result=candidate.copy();mask=~parent.molecule_id.isin(eligible)
    result.loc[mask,'smiles']=parent.loc[mask,'smiles']
    assert result.loc[mask].equals(parent.loc[mask])
    return result


def finalize(raw,prior,identity,fallback):
    """C29 first-identity dedup/backfill; invalid rows preserve the P16 parent."""
    assert raw.molecule_id.tolist()==prior.molecule_id.tolist()==fallback.molecule_id.tolist()
    assert raw.molecule_id.is_unique and prior.molecule_id.is_unique and fallback.molecule_id.is_unique
    rows=[];duplicates=0;invalid=[]
    for after,before,parent in zip(raw.itertuples(index=False),prior.itertuples(index=False),fallback.itertuples(index=False)):
        try:
            seen=set();smiles=[];raw_smiles=after.smiles.split(';')
            for s in raw_smiles+before.smiles.split(';'):
                key=identity(s)
                if not isinstance(key,str) or not key:raise ValueError('Missing official identity')
                if key not in seen:seen.add(key);smiles.append(s)
                elif s in raw_smiles:duplicates+=1
                if len(smiles)>=25:break
            if not 1<=len(smiles)<=25:raise ValueError('Invalid answer length')
            result=';'.join(smiles)
        except (ValueError,TypeError):
            result=parent.smiles;invalid.append(str(after.molecule_id))
        rows.append(dict(molecule_id=after.molecule_id,smiles=result))
    return pd.DataFrame(rows),dict(metric_identity_dedup_events=duplicates,invalid_structure_parent_fallback_ids=invalid)
