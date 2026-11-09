"""Fail-closed global single-pending and durable-intent checks for C09."""
import json,math
from pathlib import Path
from datetime import datetime,timezone

TERMINAL={'COMPLETE','ERROR','CANCELLED','CANCELED'}


def field(row,name,default=None):
    return row.get(name,default) if isinstance(row,dict) else getattr(row,name,default)


def normalize(row):
    value=field(row,'status')
    # The installed SDK omits default pending status from serialized JSON.
    status=str(getattr(value,'name',value) or 'PENDING').split('.')[-1].upper()
    raw=field(row,'publicScore',field(row,'public_score'))
    try:score=float(raw) if raw not in [None,''] else None
    except (TypeError,ValueError):score=None
    if score is not None and not math.isfinite(score):score=None
    return dict(reference=str(field(row,'ref','')),status=status,public_score=score,
        error=field(row,'errorDescription',field(row,'error_description')),
        success=status=='COMPLETE' and score is not None and not field(row,'errorDescription',field(row,'error_description')),
        description=str(field(row,'description','') or ''))


def check(rows,ledger,marker):
    assert ledger['campaign']=='C09' and ledger['maximum_pending']==1
    normalized=[normalize(row) for row in rows]
    existing=[r for r in normalized if marker in r['description']]
    assert len(existing)<=1,'Duplicate marker in account history'
    if existing:return dict(action='already_recorded',submission=existing[0])
    pending=[r for r in normalized if r['status'] not in TERMINAL]
    assert not pending,'Another leaderboard submission is pending: '+','.join(r['reference'] for r in pending)
    assert not [i for i in ledger.get('intents',[]) if not i.get('resolved')],'Ambiguous durable intent requires reconciliation'
    assert marker.startswith('C09-') and len(marker)>12,'Unique fresh C09 marker required'
    return dict(action='may_qualify',note='This check does not establish smoke/model/validation qualification')


def main():
    from kaggle.api.kaggle_api_extended import KaggleApi
    root=Path(__file__).resolve().parents[1];ledger=json.loads((root/'.local/c09-submission-ledger.json').read_text())
    api=KaggleApi();api.authenticate();rows=[normalize(r) for r in api.competition_submissions('enveda-CASMI26-molecule-id-mass-spectra')]
    path=root/'outputs/local/c09-20261109/account-status.json';path.parent.mkdir(parents=True,exist_ok=True)
    status=dict(checked_utc=datetime.now(timezone.utc).isoformat(),submissions=rows,pending=[r for r in rows if r['status'] not in TERMINAL])
    path.write_text(json.dumps(status,indent=2));print(json.dumps(dict(checked_utc=status['checked_utc'],pending=status['pending'],latest=rows[:3])))


if __name__=='__main__':main()
