"""Durable two-entry C42 guard; CPU first, its actual score, then GPU.

This command cannot qualify models. It requires a hash-bound reviewed
qualification and actual completed production smoke for the selected entry.
"""
import argparse,csv,hashlib,json,math
from pathlib import Path
from datetime import datetime,timezone
from c09_submission_guard import normalize,TERMINAL
from submit_c09_external_once import scientific_sha

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/local/c42-two-new-submissions-20261111'
LEDGER=ROOT/'.local/c42-two-new-submissions-20261111.json'
COMP='enveda-CASMI26-molecule-id-mass-spectra'


def stamp():return datetime.now(timezone.utc).isoformat()


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2),encoding='utf8');tmp.replace(path)


def register(record):
    path=ROOT/'runs.csv'
    with path.open(encoding='utf8',newline='') as stream:r=csv.DictReader(stream);fields=r.fieldnames;rows=list(r)
    run_id='C42P20' if record['trial']=='cpu' else 'C42P21'
    if any(r['id']==run_id for r in rows):return
    row={k:'' for k in fields};row.update(id=run_id,platform='kaggle-submission',kind='eval',ref=record['reference'],
        started_utc=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M'),status='running',
        purpose='New '+record['trial']+'-fitted alignment head; fixed same-formula tail and formate-compatible P16 parent',
        result='Accepted approved C42 scoring trial; actual score pending. No claimed leaderboard gain; champion0.480 retained.')
    with path.open('a',encoding='utf8',newline='') as stream:csv.DictWriter(stream,fieldnames=fields).writerow(row)


def submit(trial,dry=False):
    import msvcrt
    from kaggle.api.kaggle_api_extended import KaggleApi,ApiGetKernelRequest
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(ROOT/'.local/c42-submission.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    active_path=ROOT/'.local/c38-active-goal.json';active=json.loads(active_path.read_text())
    assert active['active'] and active['submissions_approved']==2 and active['submissions_used']<=2
    assert not active.get('scientific_gate_hold'),'Complete-parent scientific qualification is on hold for C45 repair'
    authorization=ROOT/'outputs/local/c38-source-support-20261111/authorization.json'
    qpath=OUT/'submission-qualification.json';q=json.loads(qpath.read_text())
    assert q['decision']=='execution_qualified_fixed_exploratory_pair' and q['maximum_new_accepted_submissions']==2 and q['maximum_pending']==1
    assert q['authorization_sha256']==sha(authorization) and q['champion_retained']==.480
    assert q['order']==['cpu','gpu'] and not q['claimed_public_improvement']
    for name,h in q['artifact_sha256'].items():assert sha(ROOT/name)==h,name
    e=q['trials'][trial];audit=json.loads((ROOT/e['audit_path']).read_text())
    assert audit['execution_qualified'] and audit['trial']==trial and audit['remote_source_exact']
    assert audit['local_cloud_final_lists_exact'] and audit['independent_tail_arithmetic_exact']
    assert audit['rank1_changes']==audit['candidate_membership_changes']==audit['formula_slot_changes']==audit['official_identity_duplicates']==0
    assert audit['changed_real_preview_lists']>0 and audit['real_preview_queries']==12 and audit['valid_rows']==400
    assert sha(ROOT/e['smoke_output'])==audit['csv_sha256']
    ledger=json.loads(LEDGER.read_text()) if LEDGER.exists() else dict(created_utc=stamp(),maximum_new_accepted_submissions=2,submissions=[],intents=[])
    assert ledger['maximum_new_accepted_submissions']==2 and len(ledger['submissions'])<=2
    api=KaggleApi();api.authenticate();raw=api.competition_submissions(COMP);rows=[normalize(x) for x in raw]
    marker='C42-'+trial.upper()+'-20261111'
    matches=[r for r in rows if marker in r['description']];assert len(matches)<=1
    if matches:
        record=next((r for r in ledger['submissions'] if r['trial']==trial),None)
        if record is None:
            intents=[i for i in ledger['intents'] if i['marker']==marker]
            assert len(intents)==1,'Existing account marker requires its exact durable intent'
            intent=intents[0]
            assert intent['qualification_sha256']==sha(qpath) and intent['remote_scientific_sha256']==audit['remote_scientific_sha256']
            assert intent['kernel']==e['kernel'] and intent['version']==e['version']
            record=dict(trial=trial,marker=marker,kernel=e['kernel'],version=e['version'],reconciled_utc=stamp(),
                qualification_sha256=sha(qpath),csv_sha256=audit['csv_sha256'],**matches[0]);ledger['submissions'].append(record)
        else:
            assert record['qualification_sha256']==sha(qpath) and record['csv_sha256']==audit['csv_sha256']
            record.update(matches[0],checked_utc=stamp())
        for intent in ledger['intents']:
            if intent['marker']==marker:intent.update(resolved=True,reference=record['reference'])
        if not dry:
            write(LEDGER,ledger);active['submissions_used']=len(ledger['submissions']);write(active_path,active);register(record)
        print('C42 ALREADY ACCEPTED',json.dumps(record),flush=True);return
    assert not [r for r in rows if r['status'] not in TERMINAL],'An account scoring entry is already pending'
    assert not any(not x.get('resolved') for x in ledger['intents']),'Unresolved durable submission intent; reconcile before retry'
    assert len(ledger['submissions'])<2 and not any(r['trial']==trial for r in ledger['submissions'])
    if trial=='gpu':
        first=[r for r in ledger['submissions'] if r['trial']=='cpu'];assert len(first)==1
        actual=[r for r in rows if r['reference']==str(first[0]['reference'])];assert len(actual)==1
        assert actual[0]['success'] and actual[0]['public_score'] is not None,'Observe the first actual terminal score before the second release'
        first[0].update(actual[0],checked_utc=stamp())
    else:assert not ledger['submissions']
    assert sum(x.date.date()==datetime.now(timezone.utc).date() for x in raw)<5,'Daily account submission limit'
    state=json.loads(str(api.kernels_status(e['kernel'])));assert state['status']=='COMPLETE',state
    with api.build_kaggle_client() as client:
        request=ApiGetKernelRequest();request.user_name=e['kernel'].split('/')[0];request.kernel_slug=e['kernel'].split('/')[1]
        remote=json.loads(client.kernels.kernels_api_client.get_kernel(request).to_json())
    m=remote['metadata'];assert m['currentVersionNumber']==e['version'] and m['isPrivate'] and not m['enableInternet']
    remote_sha=scientific_sha(json.loads(remote['blob']['source']))
    assert remote_sha==audit['remote_scientific_sha256']==scientific_sha(json.loads((ROOT/e['notebook_path']).read_text(encoding='utf8')))
    if dry:print('C42 READY',json.dumps(dict(trial=trial,kernel=e['kernel'],version=e['version'],submitted=False)),flush=True);return
    intent=dict(created_utc=stamp(),marker=marker,trial=trial,kernel=e['kernel'],version=e['version'],resolved=False,
        qualification_sha256=sha(qpath),remote_scientific_sha256=remote_sha)
    ledger['intents'].append(intent);write(LEDGER,ledger)
    try:
        result=api.competition_submit_code('submission.csv',marker+': Approved new '+trial+'-fitted crossmodal alignment; fixed same-formula0.20 tail, rank1/membership protected, tested formate repair. Audited source study and actual changed-path production smoke; exploratory transfer trial, no hidden-gain claim.',competition=COMP,kernel=e['kernel'],kernel_version=e['version'])
    except Exception as error:
        response=getattr(error,'response',None)
        write(OUT/(trial+'-submission-attention.json'),dict(checked_utc=stamp(),error_type=type(error).__name__,
            http_status=None if response is None else response.status_code,response_body=None if response is None else response.text,
            intent_unresolved=True,automatic_retry=False));raise
    reference=str(result.ref);assert int(reference)>0
    intent.update(resolved=True,reference=reference)
    record=dict(trial=trial,marker=marker,reference=reference,kernel=e['kernel'],version=e['version'],submitted_utc=stamp(),
        status='ACCEPTED; ACTUAL SCORE PENDING',qualification_sha256=sha(qpath),csv_sha256=audit['csv_sha256'])
    ledger['submissions'].append(record);write(LEDGER,ledger);write(OUT/(trial+'-submission.json'),record)
    active['submissions_used']=len(ledger['submissions']);write(active_path,active);register(record)
    print('C42 ACCEPTED',json.dumps(record),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('trial',choices=['cpu','gpu']);p.add_argument('--dry-run',action='store_true');a=p.parse_args();submit(a.trial,a.dry_run)
