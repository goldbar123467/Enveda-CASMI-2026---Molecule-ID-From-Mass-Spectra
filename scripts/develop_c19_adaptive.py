"""One prospective utility decision on completed C17 OOF scores; no model refit."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import json
from datetime import datetime,timezone
import numpy as np,pandas as pd
import pyarrow as pa
from precision_cache import ROOT,sha,write
from adaptive_mrr_expansion import expand
from finish_c09_confirmation import metric_vector
from tail_fusion import paired_summary


def main():
    pa.set_cpu_count(2)
    out=ROOT/'outputs/local/c19-adaptive-insertion-20261110';out.mkdir(exist_ok=True)
    assert not (out/'protocol.json').exists() and not (out/'development.json').exists()
    previous=ROOT/'outputs/local/c17-generator-fusion-20261110';base=ROOT/'outputs/local/c12-deep-candidates-20261110/all'
    cfg_path=ROOT/'configs/c19-adaptive-insertion-20261110.json';cfg=json.loads(cfg_path.read_text())
    sources=['develop_c19_adaptive.py','adaptive_mrr_expansion.py','finish_c09_confirmation.py','tail_fusion.py']
    names=['feature-freeze.json','features.npz','evidence.parquet','oof-input-freeze.json','oof-probabilities.npy','independent-oof-audit.json']
    write(out/'protocol.json',dict(prepared_utc=datetime.now(timezone.utc).isoformat(),config_sha256=sha(cfg_path),
        source_sha256={n:sha(ROOT/'scripts'/n) for n in sources},C17_input_sha256={n:sha(previous/n) for n in names},
        fresh_C13_prediction_metrics_read=False,model_refitted=False,automatic_submission=False))
    old=json.loads((previous/'oof-input-freeze.json').read_text());assert sha(previous/'oof-probabilities.npy')==old['probability_sha256']
    data=np.load(previous/'features.npz');probs=np.load(previous/'oof-probabilities.npy');pool=pd.read_parquet(previous/'evidence.parquet')
    mapping={mid:probs[i,:int(data['mask'][i].sum())] for i,mid in enumerate(data['molecule_ids'])}
    pool['probability']=np.concatenate([mapping[mid] for mid in pool.molecule_id.drop_duplicates()])
    parent=pd.read_csv(base/'parent.csv',dtype=str,keep_default_na=False);predictions,actions=expand(parent,pool,cfg)
    predictions.to_csv(out/'oof-predictions.csv',index=False);actions.to_csv(out/'oof-actions.csv',index=False)
    write(out/'input-freeze.json',dict(frozen_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(out/'protocol.json'),
        prediction_sha256=sha(out/'oof-predictions.csv'),model_refitted=False,fresh_C13_prediction_metrics_read=False,automatic_submission=False))
    labels=pd.read_csv(ROOT/'outputs/local/c10-casmi2022-20261110/labels.csv',dtype=str).set_index('molecule_id')
    identities=dict(zip(pool.smiles,pool.metric_key));baseline=metric_vector(parent,labels,identities);values=metric_vector(predictions,labels,identities)
    summary=paired_summary(baseline,values);new=int(((baseline==0)&(values>0)).sum());lost=int(((baseline>0)&(values==0)).sum())
    selected=bool(summary['delta']>0 and summary['ci95'][0]>0 and new>0)
    report=dict(completed_utc=datetime.now(timezone.utc).isoformat(),queries=160,vs_P01=summary,vs_P16=summary,
        absent_truths_recovered=new,truths_lost=lost,selected=selected,model_refitted=False,
        action_reasons={str(k):int(v) for k,v in actions.reason.value_counts().items()},
        insertion_positions={str(k):int(v) for k,v in actions[actions.position.gt(0)].position.value_counts().items()},
        all_development_previously_used=True,protocol_sha256=sha(out/'protocol.json'),input_freeze_sha256=sha(out/'input-freeze.json'),
        fresh_C13_prediction_metrics_read=False,automatic_submission=False,submission_qualified=False)
    write(out/'development.json',report)
    write(out/'selection-freeze.json',dict(selected=selected,development_sha256=sha(out/'development.json'),
        final_model_fitted=False,next_action='One fixed C17 final fit and model freeze before fresh120' if selected else 'Reject; no final fit or fresh120 read',
        fresh_C13_prediction_metrics_read=False,automatic_submission=False))
    print('C19 DEVELOPMENT',json.dumps(report),flush=True)


if __name__=='__main__':main()
