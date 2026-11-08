"""Spend exactly the one explicit C08 trial after the independent real smoke."""
import argparse,hashlib,json
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];COMP='enveda-CASMI26-molecule-id-mass-spectra';KERNEL='clarkkitchen/casmi26-p15-compatible-c08'


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2),encoding='utf-8');temp.replace(path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--version',type=int,required=True);args=parser.parse_args()
    assert not (ROOT/'.local/compute-paused.json').exists(),'Project paused'
    campaign=json.loads((ROOT/'configs/compatible-campaign-20261108.json').read_text());auth=campaign['authorization']
    assert auth['maximum_scoring_submissions']==1 and auth['marker']=='C08-P15-20261108'
    folder=ROOT/'outputs/forks/p15-compatible-c08';audit=json.loads((folder/'independent_c08_smoke_audit.json').read_text())
    assert audit['valid_rows']==400 and audit['real_molecules']==12 and audit['input_errors']==0
    assert audit['official_identity_duplicates']==audit['rank_one_changes']==audit['candidate_set_changes']==0
    assert audit['independent_tail_arithmetic'] and audit['independent_twofit_standardization_error']<1e-6
    assert audit['unchanged_p01_primary_score_error']<1e-6 and max(audit['local_Kaggle_score_max_abs_difference'].values())<1e-3
    assert audit['csv_sha256']==hashlib.sha256((folder/'submission.csv').read_bytes()).hexdigest()
    pin=json.loads((ROOT/'kaggle/forks/p15-compatible-c08/model-pin.json').read_text())
    assert audit['model']['manifest_sha256']==pin['manifest_sha256'] and audit['model']['selection_sha256']==pin['selection_sha256']
    assert audit['model']['steps']==40000 and not audit['model']['smoke']
    completed=json.loads((ROOT/campaign['training_output']/'results.json').read_text())['models']
    assert len(completed)==8 and all(m['steps']==40000 and not m['smoke'] for m in completed)
    ledger_path=ROOT/'.local/c08-submission-budget.json'
    ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else dict(authorized=1,marker=auth['marker'],submissions=[],request=auth['request'])
    assert ledger['authorized']==1 and ledger['marker']==auth['marker'] and len(ledger['submissions'])<=1
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate();submissions=api.competition_submissions(COMP)
    accepted=[s for s in submissions if auth['marker'] in (s.description or '')];assert len(accepted)<=1
    if accepted:reference=accepted[0].ref
    else:
        assert not ledger['submissions'] and not ledger.get('in_flight_intent'),'Inspect ambiguous previous API intent before any retry'
        today=datetime.now(timezone.utc).date();assert sum(s.date.date()==today for s in submissions)<5
        assert sum(str(s.status).split('.')[-1] not in ['COMPLETE','ERROR','CANCELLED'] for s in submissions)<2
        from finish_compatible_campaign import status_name
        assert status_name(api.kernels_status(KERNEL))=='COMPLETE','Only a completed real Kaggle version can be submitted'
        ledger['in_flight_intent']=dict(kernel=KERNEL,version=args.version,utc=datetime.now(timezone.utc).isoformat(),csv_sha256=audit['csv_sha256'])
        write(ledger_path,ledger)
        response=api.competition_submit_code('submission.csv',auth['marker']+': C08 selected two-fit320 scorer, conservative tail-only RRF after complete P01. Rank1/candidate set exact. Real12 Kaggle smoke, arithmetic/identity/artifact/local-Kaggle parity audited. Explicit trial; hidden improvement unproven.',competition=COMP,kernel=KERNEL,kernel_version=args.version)
        reference=response.ref
    assert reference>0
    record=dict(marker=auth['marker'],reference=reference,kernel=KERNEL,version=args.version,submitted_utc=datetime.now(timezone.utc).isoformat(),status='submitted; score pending',csv_sha256=audit['csv_sha256'])
    ledger['submissions']=[record];ledger.pop('in_flight_intent',None);write(ledger_path,ledger)
    write(ROOT/'docs/evidence/compatible-2026-11-08/p15_submission.json',record);print(json.dumps(record),flush=True)


if __name__=='__main__':main()
