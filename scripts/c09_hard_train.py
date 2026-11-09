"""C09 online hard-negative continuation of immutable C08 folds, RTX2070 only.

No external data, network, selection or submission. The pilot is a checkpointed
prefix of the same frozen schedule, so resuming does not reset its warmup.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import torch
import numpy as np
import pyarrow as pa
import argparse, gc, hashlib, json, math, random, time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from torch.nn import functional as F
from online_hard_negatives import mine
from compatible_long_run import Data, atomic_torch, SOURCES as C08_SOURCES
from precision_cache import ROOT, sha, write
from training_monitor import system_memory


def stamp(): return datetime.now(timezone.utc).isoformat()


def served_logits(z, p, positive, molecule, negative, negative_molecule, cfg):
    """Use the same raw fingerprint-head agreement term as inference."""
    primary=z @ molecule.T / cfg['temperature']
    hard=torch.einsum('bd,bkd->bk',z,negative_molecule)/cfg['temperature']
    coefficient=cfg['fp_score_weight']/cfg['fp_bits']
    primary=primary+coefficient*(p @ positive.T)
    hard=hard+coefficient*torch.einsum('bd,bkd->bk',p,negative)
    return primary,hard


def update(model,optimizer,scaler,batch,pos,neg,mask,micro,cfg,pos_weight):
    n=len(pos); optimizer.zero_grad(set_to_none=True); total=0.
    for start in range(0,n,micro):
        stop=min(start+micro,n)
        with torch.autocast('cuda',dtype=torch.float16):
            z,p=model.encode_spectra(*(x[start:stop] for x in batch))
            m=model.encode_molecules(pos)
            nm=model.encode_molecules(neg[start:stop].flatten(0,1)).reshape(stop-start,neg.shape[1],-1)
            primary,hard=served_logits(z,p,pos,m,neg[start:stop],nm,cfg)
            hard=hard.masked_fill(mask[start:stop],float('-inf'))
            logits=torch.cat([primary,hard],1)
            loss=F.cross_entropy(logits.float(),torch.arange(start,stop,device=z.device))
            loss=loss+cfg['aux_weight']*F.binary_cross_entropy_with_logits(p.float(),pos[start:stop],pos_weight=pos_weight)
            loss=loss*((stop-start)/n)
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite C09 loss')
        scaler.scale(loss).backward(); total+=float(loss.detach())
    scaler.unscale_(optimizer); norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    old=scaler.get_scale(); scaler.step(optimizer); scaler.update()
    return dict(loss=total,gradient_norm=float(norm),skipped=scaler.get_scale()<old)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,default=ROOT/'configs/c09-mined-20261109.json')
    parser.add_argument('--limit-updates',type=int,help='Stop after this many successful updates in this invocation')
    parser.add_argument('--micro-batch',type=int,default=192)
    args=parser.parse_args(); spec=json.loads(args.plan.read_text(encoding='utf-8'))
    out=ROOT/spec['output']; out.mkdir(parents=True,exist_ok=True)
    import msvcrt
    lock=(out/'trainer.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    torch.set_num_threads(2);torch.set_num_interop_threads(1);pa.set_cpu_count(2)
    assert torch.cuda.is_available() and '2070' in torch.cuda.get_device_name()
    assert args.micro_batch in [96,192,384]
    torch.cuda.set_per_process_memory_fraction(spec['gpu_memory_fraction'])
    assert not (ROOT/'.local/compute-paused.json').exists()
    sources=C08_SOURCES+['scripts/c09_train.py','scripts/c09_hard_train.py','scripts/online_hard_negatives.py','scripts/build_c09_hard_mining.py','configs/c09-mined-20261109.json']
    contract=dict(spec=spec,source_sha256={p:sha(ROOT/p) for p in sources})
    if (out/'source-contract.json').exists():assert json.loads((out/'source-contract.json').read_text())==contract,'Source/plan changed'
    else:write(out/'source-contract.json',contract)
    if not (out/'launch.json').exists():write(out/'launch.json',dict(started_utc=stamp()))
    deadline=datetime.fromisoformat(json.loads((out/'launch.json').read_text())['started_utc'])+timedelta(hours=spec['maximum_hours'])
    write(out/'manifest.json',dict(kind='compatible_scorer',spec=spec,source_sha256=contract['source_sha256'],smoke=False))
    invoked=0
    def report(**values):
        memory=system_memory(); row=dict(updated_utc=stamp(),pid=os.getpid(),deadline_utc=deadline.isoformat(),**values)
        row.setdefault('resources',{})['free_ram_gib']=memory.get('free_gib')
        write(out/'status.json',row);print(json.dumps(row),flush=True)
        if memory.get('free_gib',100)<spec['minimum_free_ram_gib']:raise RuntimeError('RAM below configured4GiB floor')
        if datetime.now(timezone.utc)>=deadline:raise RuntimeError('C09 wall budget exhausted')
        if (ROOT/'.local/compute-paused.json').exists():raise RuntimeError('Project paused')
    report(state='preparing',stage='Loading existing immutable C08 cache')
    data=Data(spec);results=[]
    for fold in spec['folds']:
        folder=out/'models'/'mined128-served'/f'fold-{fold}';folder.mkdir(parents=True,exist_ok=True)
        if (folder/'complete.json').exists():results.append(json.loads((folder/'complete.json').read_text()));continue
        base=ROOT/spec['parent_output']/'models'/spec['parent_arm']/f'fold-{fold}'/'final_model.pt'
        receipt=json.loads(base.with_name('complete.json').read_text());assert receipt['checkpoint_sha256']==sha(base) and receipt['steps']==40000
        ck=torch.load(base,map_location='cpu',weights_only=True);cfg=dict(ck['cfg'],**spec['training'],name='mined128-served')
        seed=cfg['seed']+fold;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        from model import SpectrumCandidateModel
        model=SpectrumCandidateModel(cfg).cuda().train();model.load_state_dict(ck['state_dict']);del ck
        optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'],fused=True)
        scaler=torch.amp.GradScaler('cuda');qr=np.random.default_rng(seed);nr=np.random.default_rng(seed+100)
        allowed=data.train_ids(fold);pool=np.load(ROOT/spec['cache']/f'pools-fold-{fold}.npy',mmap_mode='r')
        assert pool.shape==(data.n,512)
        pool_receipt=json.loads((ROOT/spec['cache']/f'pools-fold-{fold}.json').read_text())
        assert pool_receipt['contract']['allowed_sha256']==hashlib.sha256(allowed.tobytes()).hexdigest()
        counts=np.zeros(2048,np.float64)
        for start in range(0,len(allowed),4096):counts+=np.asarray(data.fp[allowed[start:start+4096]]).sum(0)
        pw=torch.tensor(np.clip((len(allowed)-counts)/(counts+1),1,20),device='cuda',dtype=torch.float32)
        step=attempts=skips=0;history=[];last=folder/'resume.pt'
        if last.exists():
            state=torch.load(last,map_location='cpu',weights_only=False);assert state['cfg']==cfg and state['parent_sha256']==sha(base)
            model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler'])
            step=state['step'];attempts=state['attempts'];skips=state['skips'];history=state['history']
            qr.bit_generator.state=state['query_rng'];nr.bit_generator.state=state['negative_rng'];random.setstate(state['python_rng'])
            torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng']);del state
        def save():
            atomic_torch(dict(cfg=cfg,parent_sha256=sha(base),step=step,attempts=attempts,skips=skips,history=history,
                model=model.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),query_rng=qr.bit_generator.state,
                negative_rng=nr.bit_generator.state,python_rng=random.getstate(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),last)
            write(folder/'history.json',history)
        losses=[];torch.cuda.reset_peak_memory_stats();block=time.perf_counter();block_step=step
        try:
            while step<cfg['steps']:
                ids=qr.choice(allowed,cfg['batch_size'],replace=False)
                chosen=np.argsort(nr.random((len(ids),cfg['negative_pool'])),axis=1)[:,:spec['mining']['sampled_negatives']]
                negative=np.take_along_axis(np.asarray(pool[ids]),chosen,axis=1)
                assert np.all(data.folds[negative]!=fold) and np.all(data.eligible[negative])
                batch,pos,pool_negative,pool_mask=data.batch(ids,negative,qr,'float16')
                neg,mask,mining_stats=mine(model,batch,pos,pool_negative,pool_mask,cfg,cfg['hard_negatives'])
                del pool_negative,pool_mask
                cycle_step=step%spec['cycle_steps']
                factor=min(1.,(cycle_step+1)/cfg['warmup_steps'])
                if cycle_step>=cfg['warmup_steps']:factor=.5*(1+math.cos(math.pi*(cycle_step-cfg['warmup_steps'])/(spec['cycle_steps']-cfg['warmup_steps'])))
                for group in optimizer.param_groups:group['lr']=cfg['lr']*factor
                value=update(model,optimizer,scaler,batch,pos,neg,mask,args.micro_batch,cfg,pw)
                attempts+=1;skips+=int(value['skipped']);step+=int(not value['skipped']);invoked+=int(not value['skipped']);losses.append(value['loss'])
                if skips>20 and skips/max(attempts,1)>.02:raise RuntimeError('Excess AMP skips')
                if step and step%50==0 and not value['skipped']:
                    torch.cuda.synchronize();elapsed=time.perf_counter()-block
                    row=dict(step=step,loss=float(np.mean(losses)),updates_per_second=(step-block_step)/elapsed,gradient_norm=value['gradient_norm'],**mining_stats)
                    history.append(row);write(folder/'history.json',history)
                    report(state='training',stage='C09 online hard-negative continuation',arm='mined128-served',fold=fold,step=step,loss=row['loss'],
                        resources=dict(gpu_peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,micro_batch=args.micro_batch,amp_skips=skips,updates_per_second=row['updates_per_second'],mining_gain=mining_stats['mining_gain'],selected_valid_min=mining_stats['selected_valid_min']))
                    block=time.perf_counter();block_step=step;losses=[]
                if step and step%cfg['checkpoint_interval']==0 and not value['skipped']:save()
                if step and step%spec['cycle_steps']==0 and not value['skipped']:
                    atomic_torch(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},cfg=cfg,steps=40000+step,continuation_steps=step,parent_sha256=sha(base),fold=fold,smoke=False),folder/f'milestone-{step}.pt')
                if args.limit_updates and invoked>=args.limit_updates:
                    save();report(state='pilot_checkpoint',stage='C09 pilot checkpoint ready',arm='mined128-served',fold=fold,step=step,
                        resources=dict(gpu_peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,amp_skips=skips))
                    print('DONE C09 PILOT',flush=True);return
        except BaseException:
            # Leave the previous atomic checkpoint untouched after an incomplete
            # backward/OOM. Recovery always uses a fresh process.
            raise
        save();atomic_torch(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},cfg=cfg,steps=40000+step,
            continuation_steps=step,parent_sha256=sha(base),fold=fold,smoke=False),folder/'final_model.pt')
        result=dict(fold=fold,continuation_steps=step,parent_steps=40000,parent_sha256=sha(base),checkpoint_sha256=sha(folder/'final_model.pt'),
            attempts=attempts,amp_skips=skips,training_identities=len(allowed),heldout_in_training=0,
            gpu_peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,completed_utc=stamp())
        write(folder/'complete.json',result);results.append(result);del model,optimizer,scaler;gc.collect();torch.cuda.empty_cache()
    write(out/'results.json',dict(models=results,completed_utc=stamp(),quality_gate_passed=False))
    report(state='validation_pending',stage='C09 continuation complete; full-pipeline validation required')
    print('DONE C09 TRAINING',flush=True)


if __name__=='__main__':main()
