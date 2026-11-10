"""Fixed five-fold query-held-out C17 calibration on previously used development."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import json,time
import numpy as np
import pandas as pd
import torch
import pyarrow as pa
from prepare_c17_features import OUT,POOL,CONFIG,stamp
from precision_cache import ROOT,sha,write
from expected_mrr_expansion import expand
from finish_c09_confirmation import metric_vector
from tail_fusion import paired_summary
from training_monitor import system_memory


def train(x,mask,targets,train_indices,config,seed):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    head=torch.nn.Linear(x.shape[-1],1,device=x.device)
    torch.nn.init.zeros_(head.weight);torch.nn.init.zeros_(head.bias)
    optimizer=torch.optim.Adam(head.parameters(),lr=config['learning_rate'])
    selected=torch.as_tensor(train_indices,device=x.device)
    for epoch in range(config['epochs']):
        logits=head(x[selected]).squeeze(-1).masked_fill(~mask[selected],-1e9)
        logits=torch.cat([logits,torch.zeros((len(selected),1),device=x.device)],dim=1)
        loss=torch.nn.functional.cross_entropy(logits,targets[selected])+config['l2_weights']*head.weight.square().sum()
        assert torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
    return head,float(loss.detach())


def predict(head,x,mask):
    with torch.no_grad():
        logits=head(x).squeeze(-1).masked_fill(~mask,-1e9)
        logits=torch.cat([logits,torch.zeros((len(x),1),device=x.device)],dim=1)
        return torch.softmax(logits,dim=1).cpu().numpy()


def main():
    import msvcrt
    lock=(OUT/'training.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    assert not (OUT/'development.json').exists() and not (OUT/'training-intent.json').exists(),'Inspect prior attempt; never refit blindly'
    assert not (ROOT/'.local/compute-paused.json').exists()
    torch.set_num_threads(2);torch.set_num_interop_threads(1);pa.set_cpu_count(2)
    assert torch.cuda.is_available() and '2070' in torch.cuda.get_device_name()
    torch.cuda.set_per_process_memory_fraction(.7);assert system_memory()['free_gib']>=4
    cfg=json.loads(CONFIG.read_text());protocol=json.loads((OUT/'protocol.json').read_text());freeze=json.loads((OUT/'feature-freeze.json').read_text())
    assert sha(CONFIG)==protocol['config_sha256'] and sha(OUT/'protocol.json')==freeze['protocol_sha256']
    for name,pin in protocol['source_sha256'].items():assert sha(ROOT/'scripts'/name)==pin,name
    for name,pin in protocol['input_sha256'].items():assert sha(ROOT/name)==pin,name
    assert sha(OUT/'features.npz')==freeze['features_sha256'] and sha(OUT/'evidence.parquet')==freeze['evidence_sha256']
    data=np.load(OUT/'features.npz');mids=data['molecule_ids'].tolist();folds=data['folds'];mask=data['mask']
    pool=pd.read_parquet(OUT/'evidence.parquet');parent=pd.read_csv(POOL/'parent.csv',dtype=str,keep_default_na=False)
    used=ROOT/'outputs/local/c10-casmi2022-20261110/labels.csv'
    write(OUT/'training-intent.json',dict(started_utc=stamp(),feature_freeze_sha256=sha(OUT/'feature-freeze.json'),
        protocol_sha256=sha(OUT/'protocol.json'),used_labels_sha256=sha(used),device=torch.cuda.get_device_name(),
        query_fold_split=True,all_160_including_absent_truths=True,fresh_C13_prediction_metrics_read=False,automatic_submission=False))
    labels=pd.read_csv(used,dtype=str).set_index('molecule_id');assert len(labels)==160 and labels.index.is_unique
    groups={mid:g.sort_values('prior_rank',kind='stable') for mid,g in pool.groupby('molecule_id',sort=False)}
    targets=np.full(160,128,np.int64)
    for i,mid in enumerate(mids):
        keys=groups[mid].metric_key.tolist();truth=labels.loc[mid,'key']
        if truth in keys:targets[i]=keys.index(truth)
    assert (targets==128).sum()==41 and len(set(mids))==160
    x=torch.tensor(data['features'],device='cuda');m=torch.tensor(mask,device='cuda');y=torch.tensor(targets,device='cuda')
    probs=np.full((160,129),np.nan,np.float32);audit=[];start=time.monotonic()
    for fold in range(cfg['folds']):
        train_ids=np.flatnonzero(folds!=fold);test_ids=np.flatnonzero(folds==fold)
        assert not set(train_ids)&set(test_ids) and len(train_ids)+len(test_ids)==160
        assert time.monotonic()-start<cfg['maximum_training_hours']*3600
        head,loss=train(x,m,y,train_ids,cfg,cfg['seed']+fold)
        probs[test_ids]=predict(head,x[test_ids],m[test_ids])
        torch.save(head.state_dict(),OUT/('fold'+str(fold)+'.pt'))
        audit.append(dict(fold=fold,train_molecule_ids=[mids[i] for i in train_ids],heldout_molecule_ids=[mids[i] for i in test_ids],
            successful_updates=cfg['epochs'],final_training_loss=loss,absent_train=int((targets[train_ids]==128).sum()),absent_heldout=int((targets[test_ids]==128).sum())))
        print('C17 OOF FOLD',fold,'COMPLETE',len(train_ids),len(test_ids),loss,flush=True)
    assert np.isfinite(probs).all() and np.allclose(probs.sum(axis=1),1,atol=1e-6) and (probs[:,:128][~mask]==0).all()
    np.save(OUT/'oof-probabilities.npy',probs)
    probabilities={mid:probs[i,:len(groups[mid])] for i,mid in enumerate(mids)}
    pool['probability']=np.concatenate([probabilities[mid] for mid in pool.molecule_id.drop_duplicates()])
    predictions,actions=expand(parent,pool,cfg)
    predictions.to_csv(OUT/'oof-predictions.csv',index=False);actions.to_csv(OUT/'oof-actions.csv',index=False)
    write(OUT/'oof-input-freeze.json',dict(frozen_utc=stamp(),feature_freeze_sha256=sha(OUT/'feature-freeze.json'),
        probability_sha256=sha(OUT/'oof-probabilities.npy'),prediction_sha256=sha(OUT/'oof-predictions.csv'),
        queries=160,folds=audit,fresh_C13_prediction_metrics_read=False,automatic_submission=False))
    index=pd.read_parquet(POOL/'fingerprint-smiles.parquet');identities=dict(zip(index.smiles,index.metric_key))
    baseline=metric_vector(parent,labels,identities);values=metric_vector(predictions,labels,identities)
    summary=paired_summary(baseline,values);new=int(((baseline==0)&(values>0)).sum());lost=int(((baseline>0)&(values==0)).sum())
    selected=bool(summary['delta']>0 and summary['ci95'][0]>0 and new>0)
    result=dict(completed_utc=stamp(),queries=160,vs_P01=summary,vs_P16=summary,absent_truths_recovered=new,truths_lost=lost,
        pool_absent_truths=int((targets==128).sum()),actions={str(k):int(v) for k,v in actions.reason.value_counts().items()},
        selected=selected,training_updates=cfg['folds']*cfg['epochs'],all_development_previously_used=True,
        evaluation='Each query scored only by model fitted on other query IDs. This remains development screening, not fresh confirmation.',
        oof_input_freeze_sha256=sha(OUT/'oof-input-freeze.json'),peak_Torch_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
        fresh_C13_prediction_metrics_read=False,automatic_submission=False,submission_qualified=False)
    write(OUT/'development.json',result)
    if selected:
        head,loss=train(x,m,y,np.arange(160),cfg,cfg['seed']+cfg['folds'])
        torch.save(head.state_dict(),OUT/'final-head.pt')
        write(OUT/'selection-freeze.json',dict(frozen_utc=stamp(),selected=True,final_model_sha256=sha(OUT/'final-head.pt'),
            final_training_loss=loss,final_fit_updates=cfg['epochs'],development_sha256=sha(OUT/'development.json'),
            feature_freeze_sha256=sha(OUT/'feature-freeze.json'),fresh_C13_prediction_metrics_read=False,
            automatic_submission=False,submission_qualified=False))
    else:write(OUT/'selection-freeze.json',dict(frozen_utc=stamp(),selected=False,development_sha256=sha(OUT/'development.json'),
        final_model_fitted=False,fresh_C13_prediction_metrics_read=False,automatic_submission=False))
    print('C17 DEVELOPMENT',json.dumps(result),flush=True)


if __name__=='__main__':main()
