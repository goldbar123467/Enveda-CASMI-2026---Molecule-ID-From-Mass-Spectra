"""Audit C13 channel support without dropping its documented unknown token.

Support is derived from original query metadata and exact production chemistry
code before inference, never from labels or favorable prediction outcomes.
"""
import argparse,csv,hashlib,json,re,types
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from legacy_formate_compat import patch_chem_source
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/local/c59-c13-compatibility-20261112'
FOLDER=ROOT/'outputs/forks/c59-c13-main-compatibility-v1'
QUERY=ROOT/'outputs/publish/c13-mixed-casmi2022-20261110/validation-queries.parquet'


def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as stream:
        while chunk:=stream.read(4*1024*1024):h.update(chunk)
    return h.hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2),encoding='utf8')


def support():
    assert not (ROOT/'outputs/local/c13-confirmation-20261110/evaluation-consumption.json').exists()
    pin=read(ROOT/'kaggle/forks/c59-c13-main-compatibility/compatibility-pin.json')
    previous=read(ROOT/'outputs/local/c52-unused-target-inventory-20261112/C13-query-only-compatibility-preflight.json')
    original=ROOT/'outputs/local/c37-selection-trace-20261111/pinned-public-source/chem.py'
    assert sha(QUERY)==pin['query_sha256']
    previous_query=ROOT/'outputs/forks/c13-mixed-casmi2022-validation-v2/c13_queries.parquet'
    assert sha(previous_query)==previous['query_sha256']
    assert sha(original)==previous['pinned_legacy_source_sha256']['chem.py']
    assert sha(ROOT/'scripts/legacy_formate_compat.py')==pin['compat_adapter_sha256']
    source=patch_chem_source(original.read_text(encoding='utf8'));module=types.ModuleType('c59_exact_query_support')
    exec(compile(source,str(original),'exec'),module.__dict__)
    q=pd.read_parquet(QUERY);assert len(q)==q.molecule_id.nunique()==120
    old=pd.read_parquet(previous_query).set_index('molecule_id').loc[q.molecule_id].reset_index()
    assert len(old)==120 and set(q.columns)<=set(old.columns)
    for column in q.columns:
        for a,b in zip(q[column],old[column]):
            if isinstance(a,(np.ndarray,list,tuple)):np.testing.assert_array_equal(a,b)
            elif pd.isna(a):assert pd.isna(b)
            else:assert a==b,column
    mass=np.asarray(module.neutral_mass(q.precursor_mz.to_numpy(np.float64),q.adduct.to_numpy()),np.float64)
    assert mass.shape==(120,);finite=np.isfinite(mass)
    assert finite.sum()==pin['main_scalar_preflight']['patched_finite']==119
    assert q.loc[~finite,'adduct'].tolist()==['[M+FA-H]- ']
    value=dict(source_sha256=sha(__file__),query_sha256=sha(QUERY),exact_patched_chemistry_sha256=hashlib.sha256(source.encode()).hexdigest(),
        files_sha256={str(p.relative_to(ROOT)):sha(p) for p in [QUERY,previous_query,ROOT/'outputs/local/c52-unused-target-inventory-20261112/C13-query-only-compatibility-preflight.json',original,ROOT/'scripts/legacy_formate_compat.py',ROOT/'kaggle/forks/c59-c13-main-compatibility/compatibility-pin.json']},
        all_query_ids=q.molecule_id.astype(str).tolist(),pubchem_supported_ids=q.loc[finite,'molecule_id'].astype(str).tolist(),
        pubchem_unsupported_ids=q.loc[~finite,'molecule_id'].astype(str).tolist(),queries=120,pubchem_supported=119,pubchem_unsupported=1,
        all_original_queries_retained=True,original_trailing_space_token_preserved=True,quality_labels_read=False,
        explanation='The one original unknown adduct stays in all120outputs and quality denominators. Engine2PV-v2 supports it; the exact production main/PubChem chemistry does not strip its trailing space.')
    path=OUT/'channel-support-freeze.json'
    if path.exists():assert read(path)==value,'Pre-inference support declaration changed'
    else:write(path,value)
    return value


def check_execution(events,total,pc_count):
    text='\n'.join(str(r.get('data','')) for r in events)
    failures=re.findall(r'PUBCHEM CHANNEL FAILED[^\n]*|ENGINE FAILED[^\n]*|ICE FAILED[^\n]*|GLACIER FAILED[^\n]*|poolpop failed[^\n]*',text)
    assert not failures,failures[:5]
    engine=list(map(int,re.findall(r'\bengine lists (\d+)\b',text)))
    pc=list(map(int,re.findall(r'\bpubchem lists (\d+)\b',text)))
    errors=list(map(int,re.findall(r'\berrors (\d+) \| popularity re-ranked lists',text)))
    assert engine[:2]==[total,total] and pc and all(n==pc_count for n in pc),(engine,pc,pc_count)
    assert errors and all(n==0 for n in errors),errors
    for stage in ['PubChem channel','Engine 1','candidate lists','GLACIER + ICEBERG forward models','Two-engine fusion','Metric-identity fusion','Window-40 overlay','Execution receipt']:assert stage in text,stage
    return dict(engine2_reported_query_counts=engine,pubchem_reported_query_counts=pc,main_engine_reported_errors=errors)


def check_channel_keys(all_ids,supported_ids,engine,pc,baseline_ids):
    expected=set(all_ids);supported=set(supported_ids)
    assert set(engine)==expected and set(baseline_ids)==expected,'Every original query must retain Engine2/global output coverage'
    assert set(pc)==supported,'A supported query is missing or an unsupported query was silently reinterpreted'
    return dict(all_original_queries_retained=True,exact_supported_channel_query_ids=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');args=parser.parse_args();s=support()
    if args.preflight:print('C59 PRE-INFERENCE SUPPORT FROZEN: all120retained, PC119supported, one original trailing-space fallback');return
    target=FOLDER/'independent-core-health-audit.json';assert not target.exists()
    collection=read(FOLDER/'collection-audit.json');receipt=read(FOLDER/'execution_receipt.json')
    assert collection['queries']==120 and collection['remote_source_exact'] and collection['actual_two_T4s'] and collection['baseline_replay_exact']
    assert receipt['is_full_run'] and receipt['processed_molecules']==120
    log=FOLDER/(collection['kernel'].split('/')[1]+'.log');result=check_execution(read(log),120,119)
    engine=read(FOLDER/'eng_lists.json');pc=read(FOLDER/'pc_lists.json')
    with (FOLDER/'c13_p01.csv').open(encoding='utf8',newline='') as stream:baseline=[r['molecule_id'] for r in csv.DictReader(stream)]
    assert len(baseline)==len(set(baseline))==120
    result.update(check_channel_keys(s['all_query_ids'],s['pubchem_supported_ids'],engine,pc,baseline))
    paths=[log,FOLDER/'collection-audit.json',FOLDER/'execution_receipt.json',FOLDER/'eng_lists.json',FOLDER/'pc_lists.json',FOLDER/'c13_p01.csv',OUT/'channel-support-freeze.json']
    write(target,dict(completed_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),queries=120,pubchem_supported=119,pubchem_unsupported=1,
        passed=True,quality_labels_read=False,**result,files_sha256={**s['files_sha256'],**{str(p.relative_to(ROOT)):sha(p) for p in paths}},
        limitation='All120remain in evaluation, including the unchanged unsupported input; execution completeness is not quality or target-instrument calibration.'))
    print('C59 ALL120GLOBAL / EXACT119SUPPORTED PC CHANNEL HEALTH PASS',flush=True)


if __name__=='__main__':main()
