"""Prepare or train the C08 controlled compatible-scorer experiment on RTX2070.

No network calls, model downloads, leaderboard submissions or full-data refits.
The smoke output is separate and can never resume as a production model.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import torch  # Load c10 before Arrow/NumPy DLLs on this Windows installation.
import numpy as np
import pandas as pd
import pyarrow as pa
from pathlib import Path
from datetime import datetime,timezone,timedelta
import argparse,collections,gc,hashlib,json,math,random,sys,time,traceback
from precision_cache import prepare,sha,write,ROOT
from compatible_training import accumulated_step
from training_monitor import system_memory

sys.path.insert(0,str(ROOT/'outputs/local/kit'))
from model import SpectrumCandidateModel
import features

SOURCES=['scripts/compatible_long_run.py','scripts/compatible_training.py','scripts/precision_cache.py',
         'outputs/local/kit/features.py','outputs/local/kit/model.py']


def stamp():return datetime.now(timezone.utc).isoformat()


class StopRun(Exception):pass


def atomic_torch(value,path):
    temp=path.with_suffix('.tmp');torch.save(value,temp);temp.replace(path)


class Reporter:
    def __init__(self,out,spec,deadline=None,smoke_stop_after=None):
        self.out=out;self.spec=spec;self.deadline=deadline;self.last={};self.smoke_stop_after=smoke_stop_after
    def __call__(self,**value):
        self.last.update(value)
        memory=system_memory()
        row=dict(self.last,updated_utc=stamp(),pid=os.getpid(),deadline_utc=self.deadline.isoformat() if self.deadline else None,
                 resources=dict(self.last.get('resources',{}),free_ram_gib=memory.get('free_gib')))
        write(self.out/'status.json',row);print(json.dumps(row),flush=True)
        if (ROOT/'.local/compute-paused.json').exists():raise StopRun('Project paused')
        if self.deadline and datetime.now(timezone.utc)>=self.deadline:raise StopRun('24-hour wall budget reached')
        if memory.get('free_gib',100)<self.spec['minimum_free_ram_gib']:raise StopRun('Free RAM below configured floor')
        if self.smoke_stop_after and value.get('state')=='training' and value.get('step',0)>=self.smoke_stop_after:
            raise StopRun('Deliberate smoke checkpoint/resume test')


class Data:
    def __init__(self,spec):
        self.spec=spec;self.path=ROOT/spec['cache'];base=ROOT/spec['base_cache']
        self.mol=pd.read_parquet(base/'molecules.parquet');self.n=len(self.mol)
        self.fp=np.load(base/'fingerprints.npy',mmap_mode='r');assert self.fp.shape==(self.n,2048)
        self.x=np.load(self.path/'peaks.npy',mmap_mode='r');self.meta=np.load(self.path/'metadata.npy',mmap_mode='r')
        self.lengths=np.load(self.path/'lengths.npy',mmap_mode='r');self.adduct=np.load(self.path/'adduct.npy',mmap_mode='r')
        rows=pd.read_parquet(self.path/'selected.parquet',columns=['mol_idx'])
        valid=np.load(self.path/'valid.npy',mmap_mode='r');self.row_ids=np.flatnonzero(valid)
        ids=rows.mol_idx.to_numpy()[self.row_ids];order=np.argsort(ids,kind='stable')
        self.order=self.row_ids[order];self.offset=np.r_[0,np.bincount(ids,minlength=self.n).cumsum()]
        group=np.load(ROOT/spec['split_groups']);self.groups=group['groups'][:self.n];self.folds=group['folds'][:self.n]
        assert len(self.groups)==self.n and len(self.folds)==self.n
        all_groups=group['groups'];counts=np.bincount(all_groups)
        self.eligible=(np.diff(self.offset)>0)&(counts[self.groups]==1)
        # Same-input aliases remain in the same fold and are excluded from BOTH
        # positives and negatives, so the objective never asks them to disagree.
        table=pd.DataFrame({'g':self.groups,'f':self.folds})
        assert table.groupby('g').f.nunique().max()==1,'Acquisition duplicate group crosses folds'
        write(self.path/'identity-audit.json',dict(catalogue_identities=self.n,usable_views=len(self.row_ids),
            eligible_identities=int(self.eligible.sum()),excluded_shared_input_identities=int((counts[self.groups]>1).sum()),
            zero_usable_view_identities=int((np.diff(self.offset)==0).sum()),fold_counts={str(f):int(((self.folds==f)&self.eligible).sum()) for f in np.unique(self.folds)},
            split_group_sha256=sha(ROOT/spec['split_groups']),scope='Existing identity/acquisition folds; not a new untouched validation set'))
        # Check current RDKit against the preserved catalogue before using labels.
        checked=np.random.default_rng(20261108).choice(self.n,64,replace=False)
        for i in checked:
            key,_,_,_,fp=features.structure(self.mol.smiles.iloc[i],2048)
            assert key==self.mol.key.iloc[i] and np.array_equal(fp,self.fp[i]),'Fingerprint/identity version mismatch'
        write(self.path/'fingerprint-parity.json',dict(samples=64,mismatches=0,feature_sha256=sha(ROOT/'outputs/local/kit/features.py')))

    def train_ids(self,fold):return np.flatnonzero(self.eligible&(self.folds!=fold))

    def pools(self,fold,notify):
        ids=self.train_ids(fold);count=max(a['negative_pool'] for a in self.spec['arms']);assert len(ids)>2*count
        dest=self.path/f'pools-fold-{fold}.npy';receipt=dest.with_suffix('.json')
        contract=dict(fold=int(fold),count=count,allowed_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),
                      groups_sha256=sha(ROOT/self.spec['split_groups']),builder_sha256=sha(__file__))
        previous=json.loads(receipt.read_text()) if receipt.exists() else None
        if previous and previous['contract']==contract:return np.load(dest,mmap_mode='r')
        if previous:
            before=dict(previous['contract']);after=dict(contract);before.pop('builder_sha256');after.pop('builder_sha256')
            assert before==after,'Negative pool data contract changed; use a new cache'
            # An unrelated runner change can change its whole-file hash. Reuse
            # ONLY after regenerating and comparing EVERY pool entry below.
            pool=np.load(dest,mmap_mode='r');assert pool.shape==(self.n,count)
        else:
            pool=np.lib.format.open_memmap(dest,mode='w+',dtype=np.int32,shape=(self.n,count));pool[:]=-1
        formulas=self.mol.formula.to_numpy();mass=self.mol.mass.to_numpy();assert np.isfinite(mass[ids]).all()
        siblings=collections.defaultdict(list)
        for i in ids:siblings[formulas[i]].append(int(i))
        order=ids[np.argsort(mass[ids],kind='stable')];values=mass[order];width=2*count+1
        # Chunked nearest-mass search, then same-formula-first ordering matching
        # the public kit's intent. Every element belongs to this training fold.
        for start in range(0,len(order),512):
            positions=np.arange(start,min(start+512,len(order)));query=order[positions]
            left=np.clip(positions-count,0,len(order)-width)
            indices=left[:,None]+np.arange(width)[None,:]
            distance=np.abs(values[indices]-values[positions,None]);distance[indices==positions[:,None]]=-1
            ranked=np.argsort(distance,axis=1,kind='stable')[:,1:count+1]
            nearest=order[np.take_along_axis(indices,ranked,axis=1)]
            for row,i in enumerate(query):
                same=siblings[formulas[i]]
                if len(same)>1:
                    joined=np.r_[[j for j in same if j!=i],nearest[row]]
                    _,first=np.unique(joined,return_index=True);nearest[row]=joined[np.sort(first)[:count]]
            assert np.all(nearest!=query[:,None]) and np.all(self.folds[nearest]!=fold)
            assert np.all(self.eligible[nearest])
            if previous:assert np.array_equal(pool[query],nearest),'Regenerated negative pools differ; use a new cache'
            else:pool[query]=nearest
            if start%8192==0:notify(state='preparing',stage=f'Fold {fold}: training-only hard negatives',stage_completed=int(start),stage_total=len(order))
        if not previous:pool.flush()
        digest=sha(dest)
        if previous:assert digest==previous['sha256'],'Negative pool bytes changed'
        write(receipt,dict(contract=contract,sha256=digest,training_identities=len(ids),
                          revalidated_all_entries_after_source_change=bool(previous),prior_contract=previous['contract'] if previous else None,
                          heldout_negative_identities=0,shared_input_alias_negatives=0,completed_utc=stamp()))
        return np.load(dest,mmap_mode='r')

    def batch(self,ids,neg_ids,query_rng,precision):
        rows=self.order[self.offset[ids]+(query_rng.random(len(ids))*np.diff(self.offset)[ids]).astype(np.int64)]
        peaks=np.array(self.x[rows],dtype=np.float32)
        if precision=='float16':peaks=peaks.astype(np.float16).astype(np.float32)
        x=torch.as_tensor(peaks,device='cuda')
        # Original public-kit intensity jitter, never jitter mass coordinates.
        intensity=(x[:,:,1].square()*torch.exp(.08*torch.randn_like(x[:,:,1]))).clamp_min(0)
        intensity/=intensity.max(1,keepdim=True).values.clamp_min(1e-8)
        x[:,:,1]=intensity.sqrt();x[:,:,2]=torch.log1p(100*intensity)/math.log(101)
        batch=(x,torch.as_tensor(np.array(self.lengths[rows]),device='cuda',dtype=torch.long),
               torch.as_tensor(np.array(self.meta[rows]),device='cuda'),torch.as_tensor(np.array(self.adduct[rows]),device='cuda',dtype=torch.long))
        pos=torch.as_tensor(np.array(self.fp[ids]),device='cuda').float()
        neg=torch.as_tensor(np.array(self.fp[neg_ids]),device='cuda').float()
        # Pool entries are unique, sampled without replacement. Remove overlap
        # with ALL logical-batch positives, including the other microbatches.
        mask=torch.as_tensor(np.isin(neg_ids,ids),device='cuda')
        return batch,pos,neg,mask


def train_model(data,spec,arm,fold,pool,out,report,micro,smoke):
    cfg=dict(spec['training'],**arm);folder=out/'models'/arm['name']/f'fold-{fold}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():return json.loads((folder/'complete.json').read_text())
    seed=cfg['seed']+fold;random.seed(seed);np.random.seed(seed%(2**32));torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    query_rng=np.random.default_rng(seed);negative_rng=np.random.default_rng(seed+100)
    allowed=data.train_ids(fold);model=SpectrumCandidateModel(cfg).cuda().train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'],fused=True)
    scaler=torch.amp.GradScaler('cuda');counts=np.zeros(2048,np.float64)
    for start in range(0,len(allowed),4096):counts+=np.asarray(data.fp[allowed[start:start+4096]]).sum(0)
    pos_weight=torch.as_tensor(np.clip((len(allowed)-counts)/(counts+1),1,20),device='cuda',dtype=torch.float32)
    last=folder/'resume.pt';step=attempts=skipped=0;history=[]
    if last.exists():
        state=torch.load(last,map_location='cpu',weights_only=False)
        assert state['cfg']==cfg and state['smoke']==smoke
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler'])
        step=state['step'];attempts=state['attempts'];skipped=state['skipped'];history=state['history']
        query_rng.bit_generator.state=state['query_rng'];negative_rng.bit_generator.state=state['negative_rng']
        random.setstate(state['python_rng']);torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
        del state
    def save():
        atomic_torch(dict(cfg=cfg,smoke=smoke,step=step,attempts=attempts,skipped=skipped,
            model=model.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),history=history,
            query_rng=query_rng.bit_generator.state,negative_rng=negative_rng.bit_generator.state,python_rng=random.getstate(),
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),last)
        write(folder/'history.json',history)
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter();block_step=step;losses=[]
    try:
        while step<cfg['steps']:
            if (ROOT/'.local/compute-paused.json').exists() or (report.deadline and datetime.now(timezone.utc)>=report.deadline):
                raise StopRun('Pause or wall budget reached')
            ids=query_rng.choice(allowed,cfg['batch_size'],replace=False)
            # Uniform draws without replacement using random keys. The query RNG
            # is independent, so all four arms see the same identities/views.
            keys=negative_rng.random((len(ids),cfg['negative_pool']))
            chosen=np.argsort(keys,axis=1)[:,:cfg['hard_negatives']]
            neg_ids=np.take_along_axis(np.asarray(pool[ids,:cfg['negative_pool']]),chosen,axis=1)
            batch,pos,neg,mask=data.batch(ids,neg_ids,query_rng,arm['cache_precision'])
            fraction=min(1.,(step+1)/max(1,cfg['warmup_steps']))
            if step>=cfg['warmup_steps']:
                fraction=.5*(1+math.cos(math.pi*(step-cfg['warmup_steps'])/max(1,cfg['steps']-cfg['warmup_steps'])))
            for group in optimizer.param_groups:group['lr']=cfg['lr']*fraction
            result=accumulated_step(model,optimizer,scaler,batch,pos,neg,micro,cfg,mask,pos_weight)
            attempts+=1;skipped+=int(result['amp_skipped']);step+=int(not result['amp_skipped']);losses.append(result['loss'])
            if skipped>20 and skipped/max(attempts,1)>.02:raise RuntimeError('Repeated AMP skips; inspect training numerics')
            if step and step%100==0 and not result['amp_skipped']:
                torch.cuda.synchronize();now=time.perf_counter();elapsed=now-start
                row=dict(step=step,loss=float(np.mean(losses)),seconds=elapsed,updates=step-block_step,
                    spectra_per_second=cfg['batch_size']*(step-block_step)/elapsed,micro_batch=micro,amp_skips=skipped)
                history.append(row);write(folder/'history.json',history)
                report(state='training',stage='Compatible scorer training',arm=arm['name'],fold=fold,step=step,loss=row['loss'],
                    resources=dict(gpu_peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,loss=row['loss'],micro_batch=micro,
                                   spectra_per_second=row['spectra_per_second'],amp_skips=skipped))
                start=now;block_step=step;losses=[]
            if step and step%cfg['checkpoint_interval']==0 and not result['amp_skipped']:save()
    except StopRun:
        save();raise
    # A CUDA/OOM error must resume the last atomic checkpoint in a fresh process;
    # never continue after a partial backwards/optimizer update.
    save()
    atomic_torch(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},cfg=cfg,steps=step,fold=fold,smoke=smoke),folder/'final_model.pt')
    result=dict(steps=step,attempts=attempts,amp_skips=skipped,fold=fold,arm=arm['name'],smoke=smoke,
                checkpoint_sha256=sha(folder/'final_model.pt'),training_identities=len(allowed),heldout_in_training=0,
                gpu_peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,completed_utc=stamp(),validation='pending actual-pipeline evaluation')
    write(folder/'complete.json',result);del model,optimizer,scaler;gc.collect();torch.cuda.empty_cache();return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,default=ROOT/'configs/compatible-training-20261108.json')
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--smoke-steps',type=int,default=200);parser.add_argument('--micro-batch',type=int)
    parser.add_argument('--smoke-stop-after',type=int,help='Exercise a real checkpoint/resume boundary; smoke only')
    args=parser.parse_args();spec=json.loads(args.plan.read_text());smoke=args.smoke
    assert not (smoke and args.prepare_only)
    assert smoke or args.smoke_stop_after is None,'Deliberate stops are smoke-only'
    if smoke:
        spec['output']='outputs/local/smoke-compatible-20261108';spec['run_id']='C08S'
        spec['training']=dict(spec['training'],steps=args.smoke_steps,warmup_steps=min(20,args.smoke_steps),checkpoint_interval=100)
    out=ROOT/spec['output'];out.mkdir(parents=True,exist_ok=True)
    import msvcrt
    lock=(out/'trainer.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    torch.set_num_threads(2);torch.set_num_interop_threads(1);pa.set_cpu_count(2)
    assert not (ROOT/'.local/compute-paused.json').exists(),'Project is paused'
    manifest=dict(kind='compatible_scorer',spec=spec,smoke=smoke,prepared_utc=stamp(),source_sha256={p:sha(ROOT/p) for p in SOURCES})
    write(out/'manifest.json',manifest)
    deadline=None
    if not args.prepare_only:
        assert torch.cuda.is_available() and '2070' in torch.cuda.get_device_name(),'This plan requires the RTX2070'
        torch.cuda.set_per_process_memory_fraction(spec['gpu_memory_fraction'])
        launch=out/'launch.json'
        if not launch.exists():write(launch,dict(started_utc=stamp()))
        deadline=datetime.fromisoformat(json.loads(launch.read_text())['started_utc'])+timedelta(hours=spec['maximum_hours'])
        contract=out/'source-contract.json';current=dict(source_sha256=manifest['source_sha256'],spec=spec)
        if contract.exists():assert json.loads(contract.read_text())==current,'Scientific source or plan changed; use a new reviewed run'
        else:write(contract,current)
    report=Reporter(out,spec,deadline,args.smoke_stop_after)
    try:
        report(state='preparing',stage='Checking precision cache')
        prepare(spec,report);data=Data(spec)
        pools={}
        for fold in spec['folds']:pools[fold]=data.pools(fold,report)
        write(out/'prepared.json',dict(completed_utc=stamp(),source_sha256=manifest['source_sha256'],cache=spec['cache'],
                identities=int(data.eligible.sum()),models=len(spec['arms'])*len(spec['folds']),automatic_submission=False))
        if args.prepare_only:
            report(state='prepared',stage='Ready; full training has not started',stage_completed=None,stage_total=None)
            print('DONE COMPATIBLE PREPARATION',flush=True);return
        micro=args.micro_batch or spec['micro_batch_options'][0];assert micro in spec['micro_batch_options']
        results=[]
        for fold in spec['folds']:
            for arm in spec['arms']:
                report(state='training',stage='Starting compatible scorer',arm=arm['name'],fold=fold,step=0)
                results.append(train_model(data,spec,arm,fold,pools[fold],out,report,micro,smoke))
        write(out/'results.json',dict(models=results,completed_utc=stamp(),quality_gate_passed=False,
                  note='Training only; no automatic selection, refit or submission. Failed P14 must be a negative control.'))
        report(state='complete' if smoke else 'validation_pending',stage='Smoke complete' if smoke else 'Training complete; actual-pipeline validation pending')
        print('DONE COMPATIBLE SMOKE' if smoke else 'DONE COMPATIBLE TRAINING',flush=True)
    except StopRun as exc:
        write(out/'status.json',dict(report.last,state='paused' if (ROOT/'.local/compute-paused.json').exists() else 'budget_or_resource_stop',stage=str(exc),updated_utc=stamp(),pid=os.getpid()))
        raise SystemExit(75)
    except BaseException as exc:
        write(out/'failure.json',dict(utc=stamp(),error=repr(exc),oom=isinstance(exc,torch.cuda.OutOfMemoryError),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
