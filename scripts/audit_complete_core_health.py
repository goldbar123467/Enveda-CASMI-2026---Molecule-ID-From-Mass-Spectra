"""Check actual complete-core execution, including the soft-failing PC channel.

No labels, prediction quality, cohort changes or inference are performed.
"""
import argparse,hashlib,json,re
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk:=stream.read(4*1024*1024):h.update(chunk)
    return h.hexdigest()


def check_events(events,queries):
    assert isinstance(events,list) and events,'Missing persisted execution log'
    text='\n'.join(str(r.get('data','')) for r in events if isinstance(r,dict))
    failures=re.findall(r'PUBCHEM CHANNEL FAILED[^\n]*|ENGINE FAILED[^\n]*|ICE FAILED[^\n]*|GLACIER FAILED[^\n]*|poolpop failed[^\n]*',text)
    assert not failures,'Core channel fallback/failure: '+repr(failures[:5])
    engine=list(map(int,re.findall(r'\bengine lists (\d+)\b',text)))
    pubchem=list(map(int,re.findall(r'\bpubchem lists (\d+)\b',text)))
    errors=list(map(int,re.findall(r'\berrors (\d+) \| popularity re-ranked lists',text)))
    assert len(engine)>=2 and engine[:2]==[queries,queries],('Engine2 coverage',engine,queries)
    assert pubchem and all(n==queries for n in pubchem),('PubChem coverage',pubchem,queries)
    assert errors and all(n==0 for n in errors),('Main-engine errors',errors)
    required=['PubChem channel','Engine 1','candidate lists','GLACIER + ICEBERG forward models','Two-engine fusion',
        'Metric-identity fusion','Window-40 overlay','Execution receipt']
    for stage in required:assert stage in text,('Missing completed stage marker',stage)
    return dict(engine2_reported_query_counts=engine,pubchem_reported_query_counts=pubchem,main_engine_reported_errors=errors,
        required_completed_stage_markers=required,no_soft_failed_core_channel=True,complete_query_coverage=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path);parser.add_argument('--queries',type=int,required=True);args=parser.parse_args()
    folder=args.folder.resolve();target=folder/'independent-core-health-audit.json';assert not target.exists()
    collection=folder/'collection-audit.json';c=json.loads(collection.read_text(encoding='utf8'))
    assert c['queries']==args.queries and c['remote_source_exact'] and c['actual_two_T4s'] and c['baseline_replay_exact']
    for flag in ['quality_metrics_read','quality_outcomes_read','C13_read','C13_consumed']:
        if flag in c:assert not c[flag]
    receipt=folder/'execution_receipt.json';e=json.loads(receipt.read_text(encoding='utf8'))
    assert e['is_full_run'] and e['processed_molecules']==args.queries
    log=folder/(c['kernel'].split('/')[1]+'.log');result=check_events(json.loads(log.read_text(encoding='utf8')),args.queries)
    value=dict(completed_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),queries=args.queries,passed=True,
        quality_labels_read=False,**result,files_sha256={str(p.relative_to(ROOT)):sha(p) for p in [collection,receipt,log]},
        limitation='Execution completeness only; no prediction-quality or public-improvement claim.')
    target.write_text(json.dumps(value,indent=2),encoding='utf8');print('INDEPENDENT COMPLETE CORE HEALTH PASS',args.queries,flush=True)


if __name__=='__main__':main()
