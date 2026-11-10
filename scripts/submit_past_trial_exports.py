"""Submit the two exact audited submission.csv exports, one pending globally."""
import argparse,csv,hashlib,json
from pathlib import Path
from datetime import datetime,timezone
from c09_submission_guard import check,normalize,TERMINAL
from submit_c09_external_once import scientific_sha

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/local/two-past-trials-20261110'
LEDGER=ROOT/'.local/two-past-trials-ledger-20261110.json'
COMP='enveda-CASMI26-molecule-id-mass-spectra'

def stamp():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    p=Path(p);t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2),encoding='utf-8');t.replace(p)

def register_submission(record):
    path=ROOT/'runs.csv'
    with path.open(newline='',encoding='utf-8') as stream:
        reader=csv.DictReader(stream);fields=reader.fieldnames;rows=list(reader)
    run_id='C34'+record['slot']
    if any(r['id']==run_id for r in rows):return
    row={k:'' for k in fields}
    row.update(id=run_id,platform='kaggle-submission',kind='eval',ref=str(record['reference']),
        started_utc=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M'),status='running',
        purpose='Explicit past-work '+record['slot']+' scoring trial; retained fitted two-model0.20 tail, complete P16 parent',
        result='Accepted exact source/version and submission.csv; actual score pending. Maximum two new trials and one pending globally.')
    with path.open('a',newline='',encoding='utf-8') as stream:csv.DictWriter(stream,fieldnames=fields).writerow(row)

def submit(slot,dry_run=False):
    import msvcrt
    from kaggle.api.kaggle_api_extended import KaggleApi,ApiGetKernelRequest
    from finish_compatible_campaign import status_name
    assert not (ROOT/'.local/compute-paused.json').exists(),'User paused project'
    auth=ROOT/'.local/bounded-two-submissions-20261110.json';authorization=json.loads(auth.read_text())
    assert authorization['maximum_new_submissions']==2 and authorization['maximum_pending']==1
    lock=(ROOT/'.local/c09-submission.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    qualification_path=OUT/'submission-qualification-exports.json';q=json.loads(qualification_path.read_text())
    export=q['exports'][slot];KERNEL=export['kernel'];FORK=(ROOT/export['notebook_path']).parent
    assert q['decision']=='execution_qualified_explicit_two_trials' and q['maximum_pending']==1 and q['maximum_new_submissions']==2
    assert q['authorization_sha256']==sha(auth) and q['order']==['P18','P19'] and not q['statistical_promotion']
    assert q['champion_retained']==.480
    for path,pin in q['artifact_sha256'].items():assert sha(ROOT/path)==pin,path
    audit=json.loads((ROOT/export['audit_path']).read_text());assert audit['execution_qualified']
    assert audit['independent_audit_source_sha256']==sha(ROOT/'scripts/audit_past_trial_export.py')
    assert audit['export_slot']==slot and audit['joint_prefix_exact'] and audit['export_alias_bytes_exact']
    entry=audit['recipes'][slot];assert entry['output']=='submission.csv'
    assert entry['local_cloud_final_lists_exact'] and entry['independent_tail_arithmetic_exact']
    assert entry['valid_rows']==400 and entry['real_queries']==12 and entry['changed_rows']>0 and entry['exact_fallback_rows']==388
    assert entry['official_identity_duplicates']==entry['rank_one_changes']==entry['candidate_set_changes']==0
    smoke=(ROOT/export['audit_path']).parent;assert sha(smoke/entry['output'])==entry['csv_sha256']
    marker='C09-TWO-'+slot+('-C22' if slot=='P18' else '-C18')+'-20261110'
    ledger=json.loads(LEDGER.read_text()) if LEDGER.exists() else dict(campaign='C09',maximum_pending=1,maximum_submissions=2,
        authorization_sha256=sha(auth),created_utc=stamp(),submissions=[],intents=[])
    assert ledger['maximum_submissions']==2 and ledger['authorization_sha256']==sha(auth)
    api=KaggleApi();api.authenticate();raw=api.competition_submissions(COMP);action=check(raw,ledger,marker)
    normalized=[normalize(r) for r in raw]
    if slot=='P19':
        previous=[r for r in ledger['submissions'] if r['slot']=='P18'];assert len(previous)==1,'First approved trial must precede second'
        actual=[r for r in normalized if r['reference']==str(previous[0]['reference'])];assert len(actual)==1 and actual[0]['status'] in TERMINAL
        assert actual[0]['success'] or actual[0]['error'] or actual[0]['status'] in ['ERROR','CANCELED','CANCELLED'],'First trial lacks an actual result'
        previous[0].update(actual[0],actual_checked_utc=stamp())
    if action['action']=='already_recorded':
        record=next((r for r in ledger['submissions'] if r['marker']==marker),None)
        if record is None:
            record=dict(slot=slot,marker=marker,kernel=KERNEL,version=export['version'],output=entry['output'],reconciled_utc=stamp(),**action['submission'])
            ledger['submissions'].append(record)
        for intent in ledger['intents']:
            if intent['marker']==marker:intent.update(resolved=True,reference=action['submission']['reference'])
        if not dry_run:write(LEDGER,ledger);register_submission(record)
        print(json.dumps(dict(already_recorded=True,submission=record)),flush=True);return record
    assert len(ledger['submissions'])<2 and not any(s['slot']==slot for s in ledger['submissions'])
    assert sum(r.date.date()==datetime.now(timezone.utc).date() for r in raw)<5,'Daily submission limit'
    assert status_name(api.kernels_status(KERNEL))=='COMPLETE','A real COMPLETE Kaggle smoke is mandatory'
    with api.build_kaggle_client() as client:
        req=ApiGetKernelRequest();req.user_name='clarkkitchen';req.kernel_slug=KERNEL.split('/')[1]
        remote=json.loads(client.kernels.kernels_api_client.get_kernel(req).to_json())
    m=remote['metadata'];assert m['currentVersionNumber']==export['version'] and m['isPrivate'] and not m['enableInternet']
    remote_sha=scientific_sha(json.loads(remote['blob']['source']));assert remote_sha==audit['remote_scientific_sha256']
    assert remote_sha==scientific_sha(json.loads((FORK/'notebook.ipynb').read_text(encoding='utf-8')))
    if dry_run:
        print(json.dumps(dict(ready=True,slot=slot,marker=marker,pending=0,kernel_version=export['version'],output=entry['output'],submitted=False)),flush=True);return
    intent=dict(marker=marker,slot=slot,kernel=KERNEL,version=export['version'],output=entry['output'],utc=stamp(),
        csv_sha256=entry['csv_sha256'],qualification_sha256=sha(qualification_path),remote_scientific_sha256=remote_sha,resolved=False)
    ledger['intents'].append(intent);write(LEDGER,ledger)
    try:
        response=api.competition_submit_code(entry['output'],marker+': Explicit existing-work scoring trial. Complete pinned P16 parent plus retained '
            +('C22 final55000' if slot=='P18' else 'C18 final45000')+' two-fold0.20 tail-RRF. Real12/400 Kaggle smoke, independent arithmetic/identity/fallback/local-cloud parity. Development uncertainty includes zero; no hidden-gain claim.',
            competition=COMP,kernel=KERNEL,kernel_version=export['version'])
    except Exception as exc:
        response=getattr(exc,'response',None)
        write(OUT/(slot+'-submit-error.json'),dict(checked_utc=stamp(),error_type=type(exc).__name__,
            http_status=None if response is None else response.status_code,
            response_body=None if response is None else response.text,automatic_retry=False,
            durable_intent_unresolved=True))
        raise
    reference=int(response.ref);assert reference>0
    intent.update(resolved=True,reference=reference)
    record=dict(slot=slot,marker=marker,reference=reference,kernel=KERNEL,version=export['version'],output=entry['output'],
        submitted_utc=stamp(),status='ACCEPTED; ACTUAL SCORE PENDING',csv_sha256=entry['csv_sha256'],qualification_sha256=sha(qualification_path))
    ledger['submissions'].append(record);write(LEDGER,ledger);write(OUT/(slot+'-submission.json'),record)
    register_submission(record)
    print('ACCEPTED TWO-TRIAL SLOT',json.dumps(record),flush=True);return record

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--slot',choices=['P18','P19'],required=True);p.add_argument('--dry-run',action='store_true');a=p.parse_args();submit(a.slot,a.dry_run)
