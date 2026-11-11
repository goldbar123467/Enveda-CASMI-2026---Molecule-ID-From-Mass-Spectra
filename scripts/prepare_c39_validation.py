"""Freeze controlled timsTOF source/acquisition panels before prediction outcomes.

All structure labels remain in local sidecars. This is explicitly in-corpus
validation, with reference exclusion required and pretraining exposure disclosed.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
import json,hashlib,re,zipfile
from collections import Counter
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem,DataStructs,RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from precision_cache import ROOT,sha,write,features
from build_external_cache import collision_energy
from legacy_formate_compat import normalize_legacy_adduct
from training_monitor import system_memory

CONFIG=ROOT/'configs/c39-source-supported-validation-20261111.json'


def main():
    pa.set_cpu_count(2);RDLogger.DisableLog('rdApp.*');assert system_memory()['free_gib']>=4
    cfg=json.loads(CONFIG.read_text());out=ROOT/cfg['output'];publish=ROOT/cfg['publish']
    out.mkdir(parents=True,exist_ok=True);publish.mkdir(parents=True,exist_ok=True)
    assert json.loads((ROOT/'.local/c38-active-goal.json').read_text())['active']
    assert not (out/'validation-freeze.json').exists(),'Completed panel is immutable'
    raw=ROOT/'outputs/competition-raw/train.parquet'
    mbroot=ROOT/'outputs/research/c10-massbank-current-primary-20261110'
    archive=mbroot/'MassBank-data-2026.11-pinned.zip'
    inventory=ROOT/'outputs/local/c38-source-support-20261111/massbank-latest-tims-inventory.parquet'
    protocol=dict(frozen_utc=datetime.now(timezone.utc).isoformat(),config_sha256=sha(CONFIG),preparer_sha256=sha(__file__),
        official_identity_source_sha256=sha(ROOT/'outputs/local/kit/features.py'),raw_training_sha256=sha(raw),
        massbank_archive_sha256=sha(archive),massbank_inventory_sha256=sha(inventory),
        target_metadata_sha256=sha(ROOT/'outputs/local/c37-selection-trace-20261111/unlabelled-target/test.parquet'),
        no_prediction_quality_read=True,C13_labels_read=False,study='Controlled in-corpus source/acquisition development, not unseen-by-all-models validation')
    write(out/'preparation-protocol.json',protocol)
    used=set()
    for p in ['outputs/publish/c08-validation-20261108/labels.csv','outputs/publish/c09-validation-20261109/labels.csv',
              'outputs/local/c10-casmi2022-20261110/labels.csv','outputs/local/c20-negative-development-20261110/labels.csv']:
        used.update(pd.read_csv(ROOT/p,usecols=['key']).key)
    generator=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048);cache={}
    def structure(smiles):
        if smiles not in cache:
            key,canonical,formula,mass,_=features.structure(smiles);mol=Chem.MolFromSmiles(canonical)
            cache[smiles]=dict(key=key,smiles=canonical,formula=formula,mass=mass,
                scaffold=MurckoScaffold.MurckoScaffoldSmiles(mol=mol,includeChirality=False),fp=generator.GetFingerprint(mol))
        return cache[smiles]
    def draw(source,key):return hashlib.sha256((cfg['seed']+source+':'+key).encode()).hexdigest()
    def linked(a,b):
        return a['key']==b['key'] or a['formula']==b['formula'] or bool(a['scaffold'] and a['scaffold']==b['scaffold']) or DataStructs.TanimotoSimilarity(a['fp'],b['fp'])>=cfg['chemistry_similarity_threshold']
    allowed=set(cfg['allowed_query_adducts']);queries=[];labels=[];provenance=[];selected=[]
    def add(source,partition,s,records):
        assert s['key'] not in used and records
        mid='C39Q_'+draw(source,s['key'])[:20]
        for ordinal,r in enumerate(records):
            r=dict(r);source_record=r.pop('source_record');r['molecule_id']=mid
            r['spectrum_id']='C39S_'+hashlib.sha256((source+':'+str(source_record)).encode()).hexdigest()[:20]
            queries.append(r);provenance.append(dict(molecule_id=mid,spectrum_id=r['spectrum_id'],source=source,source_record=source_record))
        labels.append(dict(molecule_id=mid,key=s['key'],smiles=s['smiles'],formula=s['formula'],mass=s['mass'],source=source,partition=partition))
        selected.append(dict(s,source=source,partition=partition,molecule_id=mid))
    mb=pd.read_parquet(inventory);mb=mb[mb.valid_identity & ~mb.in_consumed_panel].copy()
    mb['draw']=mb.metric_key.map(lambda k:draw('MassBank2026.11-timsTOF',k))
    groups=mb.sort_values('draw',kind='stable').groupby('metric_key',sort=False)
    with zipfile.ZipFile(archive) as z:
        chosen=0
        for key,g in groups:
            s=structure(g.smiles.iloc[0]);assert s['key']==key
            if key in used:continue
            records=[]
            for row in g.itertuples(index=False):
                text=z.read(row.archive_member).decode('utf8');lines=text.splitlines()
                def field(prefix):
                    values=[l[len(prefix):].strip() for l in lines if l.startswith(prefix)]
                    return values[0] if len(values)==1 else None
                adduct=normalize_legacy_adduct(field('MS$FOCUSED_ION: PRECURSOR_TYPE '))
                precursor=field('MS$FOCUSED_ION: PRECURSOR_M/Z ')
                if adduct not in allowed or precursor is None:continue
                precursor=float(precursor);charge=1
                peaks=[];active=False
                for line in lines:
                    if line.startswith('PK$PEAK:'):active=True;continue
                    if active and line.startswith('//'):break
                    if active and line.strip():
                        v=line.split()
                        try:peaks.append((float(v[0]),float(v[1])))
                        except (ValueError,IndexError):pass
                if len(peaks)<cfg['minimum_peaks']:continue
                v=np.asarray(peaks,float);good=np.isfinite(v).all(1)&(v[:,0]>0)&(v[:,1]>0);v=v[good]
                if len(v)<cfg['minimum_peaks']:continue
                v=v[np.argsort(v[:,0],kind='stable')]
                records.append(dict(source_record=row.accession,adduct=adduct,precursor_mz=precursor,
                    ionization_mode='positive' if adduct.endswith('+') else 'negative',instrument_type='timsTOF',
                    collision_energy_ev=collision_energy(field('AC$MASS_SPECTROMETRY: COLLISION_ENERGY ')),
                    ms2_mzs=v[:,0],ms2_normalized_intensities=v[:,1]/v[:,1].max(),num_peaks=len(v)))
            if records:add('MassBank2026.11-timsTOF','source_stress',s,records);chosen+=1
            if chosen==cfg['massbank_tims_source_queries']:break
    assert chosen==cfg['massbank_tims_source_queries']
    npdata=pd.read_parquet(raw,filters=[('ingest_lib','==','enveda-np-examples')])
    npdata['canonical_adduct']=npdata.adduct.map(normalize_legacy_adduct)
    npdata=npdata[npdata.canonical_adduct.isin(allowed)&npdata.num_peaks.ge(cfg['minimum_peaks'])]
    def raw_record(row,source_record):
        mz=np.asarray(row.ms2_mzs,float);it=np.asarray(row.ms2_normalized_intensities,float)
        good=np.isfinite(mz)&np.isfinite(it)&(mz>0)&(it>0);mz,it=mz[good],it[good]
        assert len(mz)>=cfg['minimum_peaks'];order=np.argsort(mz,kind='stable');mz,it=mz[order],it[order]
        return dict(source_record=source_record,adduct=normalize_legacy_adduct(row.adduct),precursor_mz=float(row.precursor_mz),
            ionization_mode=str(row.ionization_mode),instrument_type='timsTOF',collision_energy_ev=np.asarray(row.collision_energy_ev,float) if row.collision_energy_ev is not None else [],
            ms2_mzs=mz,ms2_normalized_intensities=it/it.max(),num_peaks=len(mz))
    npgroups=[]
    for rawkey,g in npdata.groupby('inchikey14',sort=False):
        s=structure(g.normalized_smiles.iloc[0])
        if s['key'] not in used:npgroups.append((draw('enveda-np-examples',s['key']),s,g))
    seen=set();chosen=0
    for _,s,g in sorted(npgroups,key=lambda x:x[0]):
        if s['key'] in seen:continue
        seen.add(s['key']);records=[raw_record(r,'enveda-np-examples:filtered-row:'+str(i)) for i,r in g.iterrows()]
        add('enveda-np-examples','source_stress',s,records);chosen+=1
        if chosen==cfg['enveda_np_source_queries']:break
    assert chosen==cfg['enveda_np_source_queries']
    print('C39 SOURCE PANELS FROZEN',len(selected),'spectra',len(queries),flush=True)
    source_selected=selected.copy()
    # Label-free source metadata fixes mode strata; structures only isolate chemistry.
    meta=pd.read_parquet(raw,columns=['ingest_lib','inchikey14','adduct','ionization_mode','num_peaks','precursor_mz'],filters=[('ingest_lib','==','enveda-180')])
    meta['canonical_adduct']=meta.adduct.map(normalize_legacy_adduct)
    meta=meta[meta.canonical_adduct.isin(allowed)&meta.num_peaks.ge(cfg['minimum_peaks'])&meta.precursor_mz.gt(0)]
    groups=meta.groupby('inchikey14',sort=False)
    modes=groups.ionization_mode.agg(lambda s:'mixed' if s.nunique()>1 else s.iloc[0])
    candidates=[]
    for mode,count in cfg['development_mode_counts'].items():
        keys=sorted(modes.index[modes.eq(mode)],key=lambda k:draw('enveda-180',k))
        candidates.extend(keys[:max(1000,count*8)])
    subset=pd.read_parquet(raw,filters=[('ingest_lib','==','enveda-180'),('inchikey14','in',candidates)])
    subset['canonical_adduct']=subset.adduct.map(normalize_legacy_adduct)
    subset=subset[subset.canonical_adduct.isin(allowed)&subset.num_peaks.ge(cfg['minimum_peaks'])&subset.precursor_mz.gt(0)]
    bykey={k:g for k,g in subset.groupby('inchikey14',sort=False)};chosen=Counter();used_official={s['key'] for s in selected};excluded=Counter()
    for rawkey in sorted(bykey,key=lambda k:draw('enveda-180',k)):
        mode=modes.loc[rawkey]
        if chosen[mode]>=cfg['development_mode_counts'][mode]:continue
        g=bykey[rawkey];s=structure(g.normalized_smiles.iloc[0])
        if s['key'] in used or s['key'] in used_official:excluded['used_or_duplicate']+=1;continue
        if any(linked(s,b) for b in source_selected):excluded['source_chemistry_overlap']+=1;continue
        records=[raw_record(r,'enveda-180:candidate-filtered-row:'+str(i)) for i,r in g.iterrows()]
        add('enveda-180','development',s,records);chosen[mode]+=1;used_official.add(s['key'])
        if sum(chosen.values())%32==0:print('C39 DEV',sum(chosen.values()),dict(chosen),flush=True)
        if dict(chosen)==cfg['development_mode_counts']:break
    assert dict(chosen)==cfg['development_mode_counts'],('Insufficient prespecified source-disjoint strata',dict(chosen))
    q=pd.DataFrame(queries);lab=pd.DataFrame(labels);assert len(lab)==352 and q.spectrum_id.is_unique and lab.molecule_id.is_unique
    # Freeze labels as a local-only sidecar; publish only unlabelled queries.
    q.to_parquet(publish/'validation-queries.parquet',index=False);lab.to_csv(out/'labels.csv',index=False)
    pd.DataFrame(provenance).to_csv(out/'provenance.csv',index=False)
    pd.DataFrame(dict(molecule_id=lab.molecule_id,smiles='CCO')).to_csv(publish/'validation-sample.csv',index=False)
    q.groupby('molecule_id').size().to_csv(out/'acquisition-counts.csv')
    # Group every source/development chemistry component; source-source links are
    # retained and disclosed rather than counted as independent identities.
    roots=list(range(len(selected)))
    def find(i):
        while roots[i]!=i:roots[i]=roots[roots[i]];i=roots[i]
        return i
    for i,a in enumerate(selected):
        for j,b in enumerate(selected[:i]):
            if linked(a,b):roots[find(i)]=find(j)
    membership=[find(i) for i in range(len(selected))]
    groupmap={r:min(selected[i]['key'] for i,t in enumerate(membership) if t==r) for r in set(membership)}
    lab['group']=[groupmap[r] for r in membership]
    assert not (set(lab.loc[lab.partition.eq('development'),'group'])&set(lab.loc[lab.partition.eq('source_stress'),'group']))
    lab.to_csv(out/'labels.csv',index=False)
    write(out/'selected-reference-exclusion-identities.json',dict(keys=sorted(set(lab.key)),scope='Global reference masking only. No per-query truth lookup or candidate insertion allowed.',chemistry_neighbor_mask_required=True))
    write(out/'validation-freeze.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(out/'preparation-protocol.json'),
        queries=352,spectra=len(q),sources=lab.source.value_counts().to_dict(),partitions=lab.partition.value_counts().to_dict(),
        chemistry_groups=int(lab.group.nunique()),development_source_groups_disjoint=True,development_mode_counts=dict(chosen),
        development_exclusions=dict(excluded),all_usable_acquisitions_in_selected_source_records_retained=True,
        files_sha256={str(p.relative_to(ROOT)):sha(p) for p in [publish/'validation-queries.parquet',publish/'validation-sample.csv',out/'labels.csv',out/'provenance.csv',out/'selected-reference-exclusion-identities.json']},
        known_pretraining_exposure=True,novel_identity_confirmation=False,quality_metrics_computed=False,reserved_C13_read=False,
        reference_mask_complete=False,new_model_fit_exclusions_complete=False,qualified_for_scoring=False))
    print('DONE C39 QUERY PANELS',len(lab),'queries',len(q),'spectra',flush=True)


if __name__=='__main__':main()
