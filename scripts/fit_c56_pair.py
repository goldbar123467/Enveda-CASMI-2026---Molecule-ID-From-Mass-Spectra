"""Fit the prospectively fixed fingerprint pair after its held-out boundary."""
import os,json,subprocess,sys,shutil
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/local/c56-bounded-semantic-heads-20261112'


def main():
    cfg=ROOT/'configs/c56-bounded-semantic-pair-20261112.json'
    boundary=json.loads((OUT/'fit-boundary-freeze.json').read_text())
    assert boundary['auxiliary_queries']==482 and boundary['new_query_identity_overlap']==boundary['original_raw_alias_survivors']==0
    assert json.loads((ROOT/'.local/c38-active-goal.json').read_text())['active']
    assert not (ROOT/'outputs/local/c13-confirmation-20261110/evaluation-consumption.json').exists()
    assert not (ROOT/'outputs/local/c53-target-holdout-20261112/evaluation-consumption.json').exists()
    # Feature preparation pins this historical, query-only context. These are
    # USED C39 encodings, not an assertion that the new482 have been encoded.
    old=ROOT/'outputs/local/c48-clean-residual-heads-20261112/query-encoding-freeze.json'
    if not (OUT/'query-encoding-freeze.json').exists():
        shutil.copy2(old,OUT/'query-encoding-freeze.json')
        (OUT/'query-encoding-context-reuse.json').write_text(json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(),
            source=str(old.relative_to(ROOT)),reused_context='Original C39 USED query encoder context only; no new482encodings or fitting targets.',
            new_source_quality_read=False,C13_read=False),indent=2),encoding='utf8')
    stages=[('features','scripts/prepare_c56_head_features.py','head-feature-freeze.json'),
        ('cpu','scripts/train_c40_cpu.py','cpu-ridge/complete.json'),('gpu','scripts/train_c48_residual_gpu.py','gpu-residual/complete.json')]
    for stage,script,finished in stages:
        if (OUT/finished).exists():continue
        record=dict(checked_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),stage=stage,config=str(cfg.relative_to(ROOT)),
            C53_quality_read=False,C13_read=False,scoring_submissions=0)
        (OUT/'fit-pair-status.json').write_text(json.dumps(record,indent=2),encoding='utf8')
        with (OUT/(stage+'-fit.log')).open('w',encoding='utf8') as log:
            subprocess.run([sys.executable,script,'--plan',str(cfg)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    (OUT/'fit-pair-status.json').write_text(json.dumps(dict(completed_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),
        stage='both_models_fitted',C53_quality_read=False,C13_read=False,scoring_submissions=0),indent=2),encoding='utf8')
    print('C56 BOTH FIXED MODELS FITTED',flush=True)


if __name__=='__main__':main()
