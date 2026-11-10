"""Select at most one winner from both fixed C28/C29 cycles; never confirm."""
import json
from datetime import datetime,timezone
from precision_cache import ROOT,sha,write

OUT=ROOT/'outputs/local/c30-global-selection-20261110'
PLAN=ROOT/'configs/c30-global-selection-20261110.json'
PATHS={'C28MV':ROOT/'outputs/local/c28-evaluation-20261110','C29':ROOT/'outputs/local/c29-public-replay-20261110'}


def main():
    assert not (OUT/'selection-freeze.json').exists(),'Outer selection already sealed'
    assert not (ROOT/'outputs/local/c13-confirmation-20261110/evaluation-consumption.json').exists()
    freeze=json.loads((OUT/'protocol.json').read_text());cfg=json.loads(PLAN.read_text())
    for n,pin in freeze['files_sha256'].items():assert sha(ROOT/n)==pin,n
    assert cfg['cycles_in_tie_order']==list(PATHS) and not cfg['automatic_confirmation'] and not cfg['automatic_submission']
    reports={};eligible=[];pins={}
    for name,path in PATHS.items():
        assert (path/'development.json').exists() and (path/'selection-freeze.json').exists(),name+' has not completed development'
        d=json.loads((path/'development.json').read_text());s=json.loads((path/'selection-freeze.json').read_text())
        assert d['queries']==240 and d['truth_absent_mrr']==0 and not d['submission_qualified']
        assert s['development_sha256']==sha(path/'development.json') and s['new_candidate_selected']==d['new_candidate_selected']
        assert not d.get('C13_confirmation_metrics_read',d.get('fresh_C13_prediction_metrics_read',False))
        reports[name]=d;pins[name]=dict(development_sha256=sha(path/'development.json'),selection_sha256=sha(path/'selection-freeze.json'))
        if d['new_candidate_selected']:
            assert all(d['comparisons'][p]['delta']>0 and d['comparisons'][p]['ci95'][0]>0 for p in ['P01','P16'])
            eligible.append(name)
    best=max((reports[n]['comparisons']['P01']['candidate_mrr25'] for n in eligible),default=None)
    winner=next((n for n in eligible if reports[n]['comparisons']['P01']['candidate_mrr25']>=best-1e-12),None)
    write(OUT/'selection-freeze.json',dict(frozen_utc=datetime.now(timezone.utc).isoformat(),selected_cycle=winner,
        new_candidate_selected=winner is not None,development_pins=pins,policy_sha256=sha(PLAN),protocol_sha256=sha(OUT/'protocol.json'),
        source_sha256=sha(__file__),C13_confirmation_metrics_read=False,automatic_confirmation=False,automatic_submission=False,
        next_action='Freeze selected all-mode model/source/recipe and reserved inference before once-only C13full120/alias/polarity gates; real Kaggle changed-path smoke and independent audits remain.' if winner else 'Reject both fixed cycles; preserve reserved120 and extra approved public trial.'))
    print('DONE C30 GLOBAL DEVELOPMENT SELECTION',winner,flush=True)


if __name__=='__main__':main()
