"""Spend only the one remaining approved F07/P14 slot after independent smoke audit."""
import argparse
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
COMP='enveda-CASMI26-molecule-id-mass-spectra'
KERNEL='clarkkitchen/casmi26-p14-frozen-channel'


def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--version',type=int,required=True);args=parser.parse_args()
    folder=ROOT/'outputs/forks/p14-frozen-channel'
    audit=json.loads((folder/'independent_frozen_smoke_audit.json').read_text())
    assert audit['valid_rows']==400 and audit['real_molecules']==12 and audit['input_errors']==0
    assert audit['official_identity_duplicates']==0 and audit['frozen_scored_molecules']>0
    assert audit['preprocess_sort']=='stable_intensity_original_index'
    comparison=json.loads((folder/'independent_p01_comparison.json').read_text())
    assert comparison['p01_primary_max_abs_error']<1e-6
    assert comparison['unchanged_p01_outsider_max_abs_error']<1e-6
    assert comparison['local_vs_kaggle_frozen_score_max_abs_error']<1e-3
    assert audit['independent_frozen_fusion_max_abs_error']<1e-6 and audit['primary_baseline_max_abs_error']<1e-6
    assert hashlib.sha256((folder/'submission.csv').read_bytes()).hexdigest()==audit['csv_sha256']
    pin=json.loads((ROOT/'kaggle/forks/p14-frozen-channel/model-pin.json').read_text())
    assert audit['model']['manifest_sha256']==pin['manifest_sha256']
    ledger_path=ROOT/'.local/f07-submission-budget.json';ledger=json.loads(ledger_path.read_text())
    assert ledger['authorized']==2 and ledger['allocation']['trained_candidate']==1
    marker=ledger['trained_marker'];assert marker=='F07-P14-20261107'
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate();submissions=api.competition_submissions(COMP)
    existing=[s for s in submissions if marker in (s.description or '')];assert len(existing)<=1
    if existing:reference=existing[0].ref
    else:
        assert len(ledger['submissions'])==1 and ledger['submissions'][0]['reference']==58105681
        assert not any(s.get('slot')=='trained_candidate' for s in ledger['submissions'])
        assert not ledger.get('in_flight_intent'),'Ambiguous earlier request: inspect accepted API submissions before retrying'
        today=datetime.now(timezone.utc).date();assert sum(s.date.date()==today for s in submissions)<5
        assert sum(str(s.status).split('.')[-1] not in ['COMPLETE','ERROR','CANCELLED'] for s in submissions)<2
        status=api.kernels_status(KERNEL);assert 'COMPLETE' in str(status),status
        message=marker+': P01 plus development-selected F07R1 frozen DreaMS/MoLFormer50% final-list channel. Reserved conditional MRR .6375 to .7423; real12 Kaggle smoke/400valid, model pins, fusion, unsupported fallback and metric dedup audited. Hidden gain unproven.'
        ledger['in_flight_intent']=dict(marker=marker,kernel=KERNEL,version=args.version,created_utc=datetime.now(timezone.utc).isoformat(),smoke_csv_sha256=audit['csv_sha256'])
        write(ledger_path,ledger)
        response=api.competition_submit_code('submission.csv',message,competition=COMP,kernel=KERNEL,kernel_version=args.version)
        reference=response.ref
    assert reference>0
    record=dict(slot='trained_candidate',marker=marker,reference=reference,kernel=KERNEL,version=args.version,file='submission.csv',
                submitted_utc=datetime.now(timezone.utc).isoformat(),status='submitted; score pending',smoke_csv_sha256=audit['csv_sha256'])
    ledger['submissions']=[s for s in ledger['submissions'] if s.get('slot')!='trained_candidate']+[record]
    ledger.pop('in_flight_intent',None);assert len(ledger['submissions'])<=ledger['authorized'];write(ledger_path,ledger)
    write(ROOT/'docs/evidence/frozen-2026-11-07/p14_submission.json',record)
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
