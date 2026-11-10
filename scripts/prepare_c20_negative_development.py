"""Prepare80 unused negative-ion development queries; never touch fresh metrics.

Labels remain local; exclude every reserved C20 identity before a fixed hash draw.
Strict source scan assignment/identity/raw audits inherited from audited C20.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import collections
import hashlib
import io
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import rdBase

from precision_cache import ROOT, sha, write, features
from casmi2022_data import parse_msp, scan_rank
from c09_spectrum_audit import signatures
from training_monitor import system_memory

OUT = ROOT / 'outputs/local/c20-negative-development-20261110'
INVENTORY = ROOT / 'outputs/local/c13-confirmation-20261110'
RESERVED = ROOT / 'outputs/local/c13-confirmation-20261110'
PUBLISH = ROOT / 'outputs/publish/c20-negative-casmi2022-20261110'
PARAMETERS = dict(positive=0, negative=80, excluded_known_errors=[81, 282, 432, 476],
    mass_assignment_da=.005, rt_assignment_minutes=.1, minimum_peaks=3,
    selection='Maximum finite positive TIC; deterministic RT/mass/name tie order; no unmatched fallback',
    draw_salt='C20-CASMI2022-unused-negative-dev-v1:', raw_signature_window='20ppm +.003Da',
    collision_energy='Unknown NaN; not imputed', stratification='80negative development after all prior/reserved exclusions; fail if insufficient')


def stamp(): return datetime.now(timezone.utc).isoformat()


def raw_overlap(queries, reference):
    precursor = pd.read_parquet(reference, columns=['precursor_mz']).precursor_mz.to_numpy(dtype=np.float64)
    order = np.argsort(precursor, kind='stable'); sorted_mz = precursor[order]
    wanted, indices = set(), [collections.defaultdict(set), collections.defaultdict(set)]
    for row in queries.itertuples(index=False):
        sig = signatures(row, features); assert sig is not None
        for i, value in enumerate(sig): indices[i][value].add(row.molecule_id)
        width = row.precursor_mz * 20e-6 + .003
        lo = np.searchsorted(sorted_mz, row.precursor_mz-width, side='left')
        hi = np.searchsorted(sorted_mz, row.precursor_mz+width, side='right')
        wanted.update(order[lo:hi].tolist())
    cross = set().union(*(ids for index in indices for ids in index.values() if len(ids) > 1))
    matches, checked, offset = [], 0, 0
    for batch in pq.ParquetFile(reference).iter_batches(batch_size=4096,
            columns=['precursor_mz', 'ms2_mzs', 'ms2_normalized_intensities'], use_threads=False):
        positions = [i for i in range(len(batch)) if offset+i in wanted]
        if positions:
            selected = batch.take(pa.array(positions)).to_pandas()
            for pos, row in zip(positions, selected.itertuples(index=False)):
                values = signatures(row, features); checked += 1
                if values is None: continue
                for i, value in enumerate(values):
                    for mid in indices[i].get(value, []):
                        matches.append(dict(molecule_id=mid, reference_row=offset+pos, signature=i))
        offset += len(batch)
        if offset % 262144 < 4096: print('C20 RAW AUDIT', offset, '/', len(precursor), 'CHECKED', checked, flush=True)
    flagged = {row['molecule_id'] for row in matches}
    pd.DataFrame(matches, columns=['molecule_id','reference_row','signature']).to_csv(OUT / 'raw-matches.csv', index=False)
    return flagged | cross, dict(raw_rows=len(precursor), candidate_rows=len(wanted), checked_rows=checked,
        reference_overlap_queries=len(flagged), cross_identity_query_aliases=len(cross), match_records=len(matches),
        labels_used=False, limitation='Bounded rounded/full and FP16top96 signatures; not universal historical-pretraining or near-duplicate exclusion')


def prior_overlap(queries):
    previous=json.loads((RESERVED/'independent-input-audit.json').read_text())['prior_query_sha256']
    reserved_queries=ROOT/'outputs/publish/c13-mixed-casmi2022-20261110/validation-queries.parquet'
    previous[str(reserved_queries.relative_to(ROOT))]=sha(reserved_queries)
    indices=[collections.defaultdict(set),collections.defaultdict(set)]
    for row in queries.itertuples(index=False):
        for i,value in enumerate(signatures(row,features)):indices[i][value].add(row.molecule_id)
    matches=[]
    for name,pin in previous.items():
        assert sha(ROOT/name)==pin
        frame=pd.read_parquet(ROOT/name)
        for row in frame.itertuples(index=False):
            values=signatures(row,features)
            if values is None:continue
            for i,value in enumerate(values):
                for mid in indices[i].get(value,[]):matches.append(dict(molecule_id=mid,prior_source=name,signature=i))
    return {m['molecule_id'] for m in matches},dict(checked_utc=stamp(),prior_query_sha256=previous,
        aliases=len({m['molecule_id'] for m in matches}),match_records=len(matches),
        prediction_metrics_read=False,reserved_C13_metrics_read=False)


def main():
    pa.set_cpu_count(2); assert rdBase.rdkitVersion == '2026.03.3' and system_memory()['free_gib'] >= 4
    assert not (PUBLISH / 'validation-contract.json').exists(), 'Completed panel is sealed'
    assert not (OUT / 'preparation-protocol.json').exists(), 'Existing attempt requires inspection before recovery'
    assert not (ROOT / '.local/compute-paused.json').exists()
    inventory = json.loads((INVENTORY / 'inventory.json').read_text())
    assert inventory['minimum_counts_sufficient'] and not inventory['prediction_files_read']
    assert sha(INVENTORY / 'identity-inventory.parquet') == inventory['inventory_sha256']
    archive = ROOT / 'outputs/research/c10-casmi2022-20261110/casmi_2022.tgz'
    raw_train = ROOT / 'outputs/competition-raw/train.parquet'
    assert sha(archive) == inventory['source_sha256'] and sha(raw_train) == inventory['raw_training_sha256']
    for name, pin in inventory['prior_label_sha256'].items(): assert sha(ROOT / name) == pin
    OUT.mkdir(exist_ok=True); PUBLISH.mkdir(exist_ok=True)
    write(OUT / 'preparation-protocol.json', dict(prepared_utc=stamp(), parameters=PARAMETERS,
        preparer_sha256=sha(__file__), parser_sha256=sha(ROOT / 'scripts/casmi2022_data.py'),
        reserved_C13_labels_sha256=sha(RESERVED / 'labels.csv'),
        reserved_C13_metrics_read=False,
        inventory_sha256=sha(INVENTORY / 'inventory.json'), source_sha256=sha(archive),
        feature_sha256=sha(ROOT / 'outputs/local/kit/features.py'),
        development_only=True, source_structures_used_only_for_exclusion=True,
        previous_C10_C12_panel_consumed=True, no_prediction_files_or_metrics_read=True,
        reserved_confirmation_used_for_selection=False, automatic_submission=False))
    assert sha(RESERVED / 'labels.csv') == 'c74a4d91eb1c9fd65855e2804036ab80238732e01e07c1da4ed9229ae152376d'
    reserved_keys = set(pd.read_csv(RESERVED / 'labels.csv',usecols=['key']).key)
    eligible = pd.read_parquet(INVENTORY / 'identity-inventory.parquet')
    eligible = eligible[eligible['mode'].eq('negative') & ~eligible.key.isin(reserved_keys)].set_index('source_number')
    assert len(eligible) >= PARAMETERS['negative'] and not set(eligible.key) & reserved_keys
    with tarfile.open(archive, 'r:gz') as data:
        meta = pd.read_csv(data.extractfile('casmi_2022/all_meta_results.csv'), header=1)
    targets = {}; counts = collections.Counter()
    for row in meta.to_dict('records'):
        number = int(row['Compound Number'])
        if number not in eligible.index: continue
        key, smiles, formula, mass, _ = features.structure(row['SMILES'])
        assert key == eligible.loc[number, 'key']
        targets[number] = dict(molecule_id='C20Q_'+hashlib.sha256((sha(archive)+':'+str(number)).encode()).hexdigest()[:20],
            key=key, smiles=smiles, formula=formula, mass=mass, mode=eligible.loc[number, 'mode'],
            source_number=number, source_file=str(row['File']), adduct=str(row['Adduct']),
            target_mz=float(row['Precursor m/z (Da)']), target_rt=float(row['RT [min]']))
    by_file = collections.defaultdict(list)
    for number, target in targets.items(): by_file[target['source_file']].append(number)
    best, seen = {}, 0
    with tarfile.open(archive, 'r:gz') as data:
        for member in data:
            if not member.isfile() or not member.name.endswith('.msp'): continue
            name = Path(member.name).stem
            if name not in by_file: continue
            for scan in parse_msp(io.TextIOWrapper(data.extractfile(member), encoding='utf-8')):
                seen += 1
                for number in by_file[name]:
                    target = targets[number]; rank = scan_rank(scan, target['target_mz'], target['target_rt'])
                    if rank is not None and (number not in best or rank < best[number][0]): best[number] = rank, scan
    rows, labels, provenance = [], [], []
    for number, target in targets.items():
        if number not in best: counts['no_matching_scan'] += 1; continue
        rank, scan = best[number]; peaks = np.asarray(scan['peaks'], dtype=np.float64)
        peaks = peaks[np.isfinite(peaks).all(1) & (peaks[:,0] > 0) & (peaks[:,1] > 0)]
        peaks = peaks[np.argsort(peaks[:,0], kind='stable')]
        if len(peaks) < 3: counts['too_few_peaks'] += 1; continue
        mz, intensity = peaks[:,0].astype(np.float32), (peaks[:,1]/peaks[:,1].max()).astype(np.float32)
        if features.peaks(mz, intensity, scan['precursor_mz'], 96)[1] < 3:
            counts['too_few_model_peaks'] += 1; continue
        mid = target['molecule_id']; sid = 'C20S_'+hashlib.sha256((target['source_file']+':'+scan['name']).encode()).hexdigest()[:20]
        rows.append(dict(molecule_id=mid, spectrum_id=sid, adduct=target['adduct'], precursor_mz=scan['precursor_mz'],
            ionization_mode=target['mode'], instrument_type='FT', collision_energy_ev=float('nan'),
            ms2_mzs=mz, ms2_normalized_intensities=intensity, num_peaks=len(mz)))
        labels.append({k:target[k] for k in ['molecule_id','key','smiles','formula','mass','mode']})
        provenance.append(dict(molecule_id=mid,spectrum_id=sid,source_number=number,source_file=target['source_file'],
            scan_name=scan['name'],rt=scan['rt'],target_rt=target['target_rt'],target_mz=target['target_mz'],
            observed_mz=scan['precursor_mz'],tic=-rank[0],license='CC-BY-4.0'))
    queries = pd.DataFrame(rows); labels = pd.DataFrame(labels); provenance = pd.DataFrame(provenance)
    assert labels.key.is_unique and queries.spectrum_id.is_unique
    assert not {'key','smiles','formula','mass','source_number'} & set(queries.columns)
    flagged, audit = raw_overlap(queries, raw_train)
    prior_flagged, prior_audit = prior_overlap(queries)
    flagged |= prior_flagged
    write(OUT / 'prior-signature-audit.json', prior_audit)
    labels = labels[~labels.molecule_id.isin(flagged)].copy()
    labels['draw'] = labels.key.map(lambda key:hashlib.sha256((PARAMETERS['draw_salt']+key).encode()).hexdigest())
    available = labels['mode'].value_counts().to_dict()
    write(OUT / 'scan-audit.json', dict(completed_utc=stamp(), counts=dict(counts), scans_read=seen,
        eligible_after_scan_and_raw_audit=available, raw_audit=audit, prediction_metrics_read=False))
    assert available.get('negative',0) >= 80, 'Insufficient development identities; never resample on performance'
    labels = labels[labels['mode'].eq('negative')].sort_values('draw',kind='stable').head(80).reset_index(drop=True)
    labels['partition'] = 'development'; ids = labels.molecule_id.tolist(); order = {mid:i for i,mid in enumerate(ids)}
    queries = queries[queries.molecule_id.isin(ids)].copy(); queries['order'] = queries.molecule_id.map(order)
    queries = queries.sort_values('order').drop(columns='order').reset_index(drop=True)
    assert len(queries) == len(labels) == 80 and not set(labels.key) & reserved_keys
    labels.drop(columns='draw').to_csv(OUT / 'labels.csv',index=False)
    provenance[provenance.molecule_id.isin(ids)].to_csv(OUT / 'provenance.csv',index=False)
    queries.to_parquet(PUBLISH / 'validation-queries.parquet',index=False)
    pd.DataFrame(dict(molecule_id=ids,smiles='CCO')).to_csv(PUBLISH / 'validation-sample.csv',index=False)
    write(OUT / 'partitions.json',dict(development=ids,negative=ids))
    report = dict(prepared_utc=stamp(),version=1,queries=80,spectra=80,development=80,confirmation=0,
        modes={'positive':0,'negative':80},source_doi='10.5281/zenodo.8371051',declared_license='CC-BY-4.0',
        preparation_protocol_sha256=sha(OUT / 'preparation-protocol.json'),inventory_sha256=sha(INVENTORY / 'inventory.json'),
        scan_audit_sha256=sha(OUT / 'scan-audit.json'),raw_training_sha256=inventory['raw_training_sha256'],
        labels_sha256=sha(OUT / 'labels.csv'),partitions_sha256=sha(OUT / 'partitions.json'),parameters=PARAMETERS,
        source_structures_used_only_for_exclusions=True,prediction_metrics_computed=False,query_labels_visible_to_pipeline=False,
        truth_injection=False,truth_absent_mrr=0,recipe_selected=False,automatic_submission=False,
        files={n:sha(PUBLISH/n) for n in ['validation-queries.parquet','validation-sample.csv']},
        reserved_C13_keys_excluded=True,reserved_C13_metrics_read=False,prior_signature_audit_sha256=sha(OUT / 'prior-signature-audit.json'),
        limitation='Development-only negative CASMI2022,FT/HCD source metadata,unknownCE,one maximum-TIC MS2 per identity. No universal historical-pretraining exclusion or hidden-score calibration guarantee.')
    write(PUBLISH / 'validation-contract.json',report);write(OUT / 'preparation-report.json',report)
    write(PUBLISH / 'dataset-metadata.json',dict(id='clarkkitchen/casmi26-c20-negative-casmi2022-20261110',
        title='CASMI26 C20 negative development queries',licenses=[dict(name='CC-BY-4.0')]))
    (PUBLISH / 'README.md').write_text('Private query-only CASMI2022 development panel,80 negative identities. No labels or partitions mounted. Source: Young, Wang and Roest, Public Data files for MassFormer, https://doi.org/10.5281/zenodo.8371051 , CC-BY-4.0; original UC Davis CASMI2022 data. Strict mass/RT maximum-TIC assignment; known errors, all prior training/panel/reserved C13 identities and bounded spectrum aliases excluded. Collision energy unknown. Development-only, never call this fresh confirmation. The separate C13mixed120 remains reserved; no metrics from it are read.\n',encoding='utf-8')
    print('DONE C20 NEGATIVE DEVELOPMENT PREPARATION',json.dumps(report),flush=True)


if __name__ == '__main__': main()
