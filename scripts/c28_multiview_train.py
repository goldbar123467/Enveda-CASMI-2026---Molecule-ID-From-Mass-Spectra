"""C28 same-polarity two-view continuation of immutable C08 folds, RTX2070 only.

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
from compatible_long_run import Data, atomic_torch, SOURCES as C08_SOURCES
from precision_cache import ROOT, sha, write
from training_monitor import system_memory


def stamp(): return datetime.now(timezone.utc).isoformat()


from multiview_training import ViewSampler,update


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,default=ROOT/'configs/c28-multiview-20261110.json')
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
    sources=C08_SOURCES+['scripts/c09_control_train.py','scripts/c28_multiview_train.py','scripts/multiview_training.py','configs/c28-multiview-20261110.json','configs/c28-evaluation-20261110.json']
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
        if datetime.now(timezone.utc)>=deadline:raise RuntimeError('C28 wall budget exhausted')
        if (ROOT/'.local/compute-paused.json').exists():raise RuntimeError('Project paused')
    report(state='preparing',stage='Loading existing immutable C08 cache')
    data=Data(spec);results=[]
    for fold in spec['folds']:
        folder=out/'models'/'multi-view'/f'fold-{fold}';folder.mkdir(parents=True,exist_ok=True)
        if (folder/'complete.json').exists():results.append(json.loads((folder/'complete.json').read_text()));continue
        base=ROOT/spec['parent_output']/'models'/spec['parent_arm']/f'fold-{fold}'/'final_model.pt'
        receipt=json.loads(base.with_name('complete.json').read_text());assert receipt['checkpoint_sha256']==sha(base) and receipt['steps']==40000
        ck=torch.load(base,map_location='cpu',weights_only=True);cfg=dict(ck['cfg'],**spec['training'],name='multi-view')
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
        sampler=ViewSampler(data,seed+280000,spec['multi_view']['probability'])
        if last.exists():
            state=torch.load(last,map_location='cpu',weights_only=False);assert state['cfg']==cfg and state['parent_sha256']==sha(base)
            model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler'])
            step=state['step'];attempts=state['attempts'];skips=state['skips'];history=state['history']
            qr.bit_generator.state=state['query_rng'];nr.bit_generator.state=state['negative_rng'];random.setstate(state['python_rng'])
            torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng']);sampler.load_state_dict(state['view_sampler']);del state
        def save():
            atomic_torch(dict(cfg=cfg,parent_sha256=sha(base),step=step,attempts=attempts,skips=skips,history=history,
                model=model.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),query_rng=qr.bit_generator.state,view_sampler=sampler.state_dict(),
                negative_rng=nr.bit_generator.state,python_rng=random.getstate(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),last)
            write(folder/'history.json',history)
        losses=[];torch.cuda.reset_peak_memory_stats();block=time.perf_counter();block_step=step
        try:
            while step<cfg['steps']:
                ids=qr.choice(allowed,cfg['batch_size'],replace=False)
                chosen=np.argsort(nr.random((len(ids),cfg['negative_pool'])),axis=1)[:,:cfg['hard_negatives']]
                negative=np.take_along_axis(np.asarray(pool[ids]),chosen,axis=1)
                assert np.all(data.folds[negative]!=fold) and np.all(data.eligible[negative])
                batch,pos,neg,mask=sampler.batch(ids,negative,qr,'float16')
                factor=min(1.,(step+1)/cfg['warmup_steps'])
                if step>=cfg['warmup_steps']:factor=.5*(1+math.cos(math.pi*(step-cfg['warmup_steps'])/(cfg['steps']-cfg['warmup_steps'])))
                for group in optimizer.param_groups:group['lr']=cfg['lr']*factor
                value=update(model,optimizer,scaler,batch,pos,neg,mask,args.micro_batch,cfg,pw)
                attempts+=1;skips+=int(value['skipped']);step+=int(not value['skipped']);invoked+=int(not value['skipped']);losses.append(value['loss'])
                if skips>20 and skips/max(attempts,1)>.02:raise RuntimeError('Excess AMP skips')
                if step and step%50==0 and not value['skipped']:
                    torch.cuda.synchronize();elapsed=time.perf_counter()-block
                    row=dict(step=step,loss=float(np.mean(losses)),updates_per_second=(step-block_step)/elapsed,gradient_norm=value['gradient_norm'])
                    history.append(row);write(folder/'history.json',history)
                    report(state='training',stage='C28 same-polarity two-view continuation',arm='multi-view',fold=fold,step=step,loss=row['loss'],
                        resources=dict(gpu_peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,micro_batch=args.micro_batch,amp_skips=skips,updates_per_second=row['updates_per_second']))
                    block=time.perf_counter();block_step=step;losses=[]
                if step and step%cfg['checkpoint_interval']==0 and not value['skipped']:save()
                if args.limit_updates and invoked>=args.limit_updates:
                    save();report(state='pilot_checkpoint',stage='C28 pilot checkpoint ready',arm='multi-view',fold=fold,step=step,
                        resources=dict(gpu_peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,amp_skips=skips))
                    print('DONE C28 PILOT',flush=True);return
        except BaseException:
            # Leave the previous atomic checkpoint untouched after an incomplete
            # backward/OOM. Recovery always uses a fresh process.
            raise
        save();atomic_torch(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},cfg=cfg,steps=40000+step,
            continuation_steps=step,parent_sha256=sha(base),fold=fold,smoke=False),folder/'final_model.pt')
        result=dict(fold=fold,continuation_steps=step,parent_steps=40000,parent_sha256=sha(base),checkpoint_sha256=sha(folder/'final_model.pt'),
            attempts=attempts,amp_skips=skips,training_identities=len(allowed),heldout_in_training=0,view_sampling_counts=dict(sampler.counts),
            gpu_peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,completed_utc=stamp())
        write(folder/'complete.json',result);results.append(result);del model,optimizer,scaler;gc.collect();torch.cuda.empty_cache()
    write(out/'results.json',dict(models=results,completed_utc=stamp(),quality_gate_passed=False))
    report(state='validation_pending',stage='C28 continuation complete; full-pipeline validation required')
    print('DONE C28 TRAINING',flush=True)


if __name__=='__main__':main()
