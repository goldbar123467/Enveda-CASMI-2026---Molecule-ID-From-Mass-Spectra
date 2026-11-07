"""Resumable DreaMS/MoLFormer feature caching and candidate-based head training.

This is an adapted experiment, not a reproduction of the MSAlign benchmark.
Only positive singly charged spectra <=1000Th use the new encoder; production
must retain the known baseline for unsupported queries. No automatic submission.
"""
import argparse
import ctypes
from datetime import datetime, timedelta, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('NUMBA_NUM_THREADS', '2')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import torch  # Load Torch DLLs before the legacy NumPy/Arrow stack on Windows.
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from torch.nn import functional as F
from frozen_model import ProjectionRanker

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'outputs/local/kit'))
import features as legacy_features


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(4*1024*1024): digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8'); temp.replace(path)


def save_torch(path, value):
    temp = path.with_suffix('.tmp'); torch.save(value, temp); temp.replace(path)


def free_ram():
    class Memory(ctypes.Structure):
        _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [(n, ctypes.c_ulonglong) for n in
            ['totalPhysical','availablePhysical','totalPage','availablePage','totalVirtual','availableVirtual','availableExtended']]
    m=Memory();m.length=ctypes.sizeof(m)
    if os.name == 'nt' and ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)): return m.availablePhysical/2**30
    return 100.


class Run:
    def __init__(self, spec):
        self.spec=spec;self.out=ROOT/spec['output'];self.out.mkdir(parents=True,exist_ok=True)
        launch=self.out/'launch.json'
        if not launch.exists(): save_json(launch, dict(started_utc=datetime.now(timezone.utc).isoformat()))
        self.started=datetime.fromisoformat(json.loads(launch.read_text())['started_utc'])
        self.deadline=self.started+timedelta(hours=spec['maximum_hours'])
        self.encoder=ROOT/spec['encoders']
        self.encoder_manifest=json.loads((self.encoder/'manifest.json').read_text())
        for item in self.encoder_manifest['files']:
            assert sha(self.encoder/item['file'])==item['sha256'],item['file']
        contract=dict(kind='frozen_retrieval',spec=spec,encoder_manifest_sha256=sha(self.encoder/'manifest.json'),
                      source_sha256={p:sha(ROOT/p) for p in ['scripts/frozen_retrieval.py','scripts/frozen_model.py','outputs/local/kit/features.py']})
        path=self.out/'manifest.json'
        if path.exists(): assert json.loads(path.read_text())==contract,'Run contract changed; choose a new output directory'
        else: save_json(path,contract)

    def status(self, stage, **extra):
        if (ROOT/'.local/compute-paused.json').exists(): raise RuntimeError('Project paused')
        if datetime.now(timezone.utc)>=self.deadline: raise TimeoutError('Global 24-hour budget reached; resumable artifacts retained')
        assert free_ram()>=self.spec['minimum_free_ram_gib'],'RAM reserve reached'
        value=dict(state='training',kind='frozen_retrieval',stage=stage,pid=os.getpid(),
                   updated_utc=datetime.now(timezone.utc).isoformat(),deadline_utc=self.deadline.isoformat(),
                   resources=dict(gpu_peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,free_ram_gib=free_ram()),**extra)
        save_json(self.out/'status.json',value)


def prepare(run):
    out=run.out;spec=run.spec
    if (out/'prepared.json').exists(): return
    run.status('preparing precise spectra')
    mol=pd.read_parquet(ROOT/spec['base_cache']/'molecules.parquet')
    lookup=dict(zip(mol.key,range(len(mol))))
    cols=['inchikey14','adduct','precursor_mz','ionization_mode','num_peaks','ingest_lib']
    raw=pd.read_parquet(ROOT/spec['raw_training'],columns=cols)
    mapped=raw.inchikey14.map(lookup)
    allowed={'[M+H]+','[M+Na]+','[M+K]+','[M+NH4]+','[M-H2O+H]+','[M-2H2O+H]+'}
    valid=mapped.notna() & raw.adduct.isin(allowed) & raw.ionization_mode.str.lower().eq('positive') & raw.precursor_mz.le(1000) & raw.precursor_mz.gt(0) & raw.num_peaks.ge(3)
    frame=raw.loc[valid].copy();frame['mol_idx']=mapped[valid].astype(np.int32);frame['row']=frame.index.to_numpy()
    # Deterministically sample acquisition diversity without over-weighting compounds with many scans.
    frame['rank']=(frame.row.to_numpy(np.uint64)*np.uint64(11400714819323198485)) ^ np.uint64(20261107)
    distinct=frame.sort_values('rank',kind='stable').drop_duplicates(['mol_idx','adduct','ingest_lib'])
    picked=pd.concat([distinct,frame.sort_values('rank',kind='stable')]).drop_duplicates('row').groupby('mol_idx',sort=False).head(spec['spectra_per_molecule'])
    groups=np.load(ROOT/spec['split_groups'],allow_pickle=False)
    folds=groups['folds'][:len(mol)]
    old=set(np.load(ROOT/'outputs/local/long-20261106/evaluation.npz',allow_pickle=False)['keys'])
    old.update((ROOT/'outputs/local/seeds/np_holdout_keys.txt').read_text().split())
    eligible=np.unique(picked.mol_idx.to_numpy())
    group_sizes=np.bincount(groups['groups'])
    eligible=eligible[group_sizes[groups['groups'][eligible]]==1]
    train=eligible[(folds[eligible]!=0)&~mol.key.iloc[eligible].isin(old).to_numpy()]
    held=[int(i) for i in eligible if folds[i]==0 and mol.key.iloc[i] not in old]
    held.sort(key=lambda i:hashlib.sha256(('F07-fresh:'+mol.key.iloc[i]).encode()).digest())
    n=spec['validation_molecules'];assert len(held)>=2*n
    dev=np.array(held[:n]);confirm=np.array(held[n:2*n]);eval_ids=np.r_[dev,confirm]
    assert not set(train)&set(eval_ids)
    assert not set(groups['groups'][train])&set(groups['groups'][eval_ids])
    pilot=sorted(train,key=lambda i:hashlib.sha256(('F07-pilot:'+mol.key.iloc[i]).encode()).digest())[:spec['pilot_molecules']]
    keep=np.r_[train,eval_ids]
    picked=picked[picked.mol_idx.isin(keep)].sort_values('row',kind='stable').reset_index(drop=True)
    mol.to_parquet(out/'molecules.parquet',index=False)
    picked[['row','mol_idx','adduct','precursor_mz','ingest_lib']].to_parquet(out/'selected.parquet',index=False)
    np.savez(out/'split.npz',train=train,development=dev,confirmation=confirm,pilot=np.array(pilot),folds=folds)
    del raw,frame,distinct;mapped=None;gc.collect()
    total=len(picked)
    peaks=np.lib.format.open_memmap(out/'spectra.npy',mode='w+',dtype=np.float32,shape=(total,101,2))
    metadata=np.lib.format.open_memmap(out/'metadata.npy',mode='w+',dtype=np.float32,shape=(total,4))
    adduct=np.lib.format.open_memmap(out/'adduct.npy',mode='w+',dtype=np.int64,shape=(total,))
    valid_rows=np.zeros(total,dtype=bool)
    from dreams.utils.data import SpectrumPreprocessor
    from dreams.utils.dformats import DataFormatA
    processor=SpectrumPreprocessor(DataFormatA(),n_highest_peaks=100)
    raw_to_selected=np.full(pq.ParquetFile(ROOT/spec['raw_training']).metadata.num_rows,-1,dtype=np.int32)
    raw_to_selected[picked.row.to_numpy()]=np.arange(total,dtype=np.int32)
    offset=0;written=0;trimmed=0
    columns=['ms2_mzs','ms2_normalized_intensities','precursor_mz','adduct','ionization_mode','collision_energy_ev']
    for batch in pq.ParquetFile(ROOT/spec['raw_training']).iter_batches(batch_size=2048,columns=columns):
        positions=raw_to_selected[offset:offset+len(batch)];mask=positions>=0
        if mask.any():
            subset=batch.filter(pa.array(mask)).to_pandas()
            for dst,row in zip(positions[mask],subset.itertuples(index=False)):
                if len(row.ms2_mzs)!=len(row.ms2_normalized_intensities):continue
                raw_peaks=np.stack([row.ms2_mzs,row.ms2_normalized_intensities],axis=1)
                trimmed+=int((raw_peaks[:,0]>1000).any())
                good=np.isfinite(raw_peaks).all(1)&(raw_peaks[:,0]>0)&(raw_peaks[:,0]<=1000)&(raw_peaks[:,1]>0)
                raw_peaks=raw_peaks[good]
                if len(raw_peaks)<3:continue
                peaks[dst]=processor(raw_peaks,row.precursor_mz,high_form=True)
                metadata[dst],adduct[dst]=legacy_features.metadata(row.precursor_mz,row.adduct,row.ionization_mode,row.collision_energy_ev)
                valid_rows[dst]=True;written+=1
        offset+=len(batch)
        if offset%32768<2048: run.status('preparing precise spectra',stage_completed=offset,stage_total=len(raw_to_selected))
    assert written>=.99*total,'More than1% unusable selected spectra; inspect intake'
    picked['valid']=valid_rows;picked[['row','mol_idx','adduct','precursor_mz','ingest_lib','valid']].to_parquet(out/'selected.parquet',index=False)
    usable=np.unique(picked.loc[valid_rows,'mol_idx'].to_numpy())
    train=train[np.isin(train,usable)];dev=dev[np.isin(dev,usable)];confirm=confirm[np.isin(confirm,usable)]
    pilot=np.array(pilot)[np.isin(pilot,train)]
    assert min(len(dev),len(confirm))>=min(200,spec['validation_molecules'])
    np.savez(out/'split.npz',train=train,development=dev,confirmation=confirm,pilot=pilot,folds=folds)
    for value in [peaks,metadata,adduct]:value.flush()
    save_json(out/'prepared.json',dict(molecules=len(mol),selected_spectra=total,usable_spectra=written,training_molecules=len(train),
              fresh_development=len(dev),fresh_confirmation=len(confirm),raw_train_sha256=sha(ROOT/spec['raw_training']),
              mol_id_order_sha256=sha(out/'molecules.parquet'),selection_sha256=sha(out/'selected.parquet'),
              supported='singly positive allowed adducts, precursor and retained fragments <=1000Th, >=3 usable peaks',
              spectra_with_above_range_peaks_removed=trimmed,truth_forced_diagnostic=True))


def encode_shards(run,name,ids,dimension,encode):
    path=run.out/f'{name}.npy'
    size=len(pd.read_parquet(run.out/'molecules.parquet')) if name=='molecule_embeddings' else len(pd.read_parquet(run.out/'selected.parquet'))
    vectors=np.lib.format.open_memmap(path,mode='r+' if path.exists() else 'w+',dtype=np.float16,shape=(size,dimension))
    directory=run.out/(name+'-shards');directory.mkdir(exist_ok=True)
    shard=run.spec['encoding_shard_size'];batch=run.spec['encoding_batch_size']
    for start in range(0,len(ids),shard):
        rows=np.asarray(ids[start:start+shard]);key=hashlib.sha256(rows.astype('<i8').tobytes()).hexdigest()[:20]
        target=directory/(key+'.npz')
        if target.exists():
            saved=np.load(target,allow_pickle=False);assert np.array_equal(saved['rows'],rows)
            values=saved['values'];assert values.shape==(len(rows),dimension) and np.isfinite(values).all()
        else:
            parts=[]
            for index in range(0,len(rows),batch):
                values=encode(rows[index:index+batch]);assert values.shape==(len(rows[index:index+batch]),dimension)
                assert np.isfinite(values).all()
                parts.append(values.astype(np.float16))
                if index%(batch*16)==0: run.status('encoding '+name,stage_completed=start+index,stage_total=len(ids))
            values=np.concatenate(parts)
            temp=target.with_suffix('.tmp')
            with temp.open('wb') as stream:np.savez(stream,rows=rows,values=values)
            temp.replace(target)
        vectors[rows]=values;vectors.flush()
        run.status('encoding '+name,stage_completed=min(start+shard,len(ids)),stage_total=len(ids))
    return vectors


def molecule_cache(run):
    if (run.out/'molecules_encoded.json').exists():return
    if run.spec.get('reuse_molecule_cache_from'):
        source=ROOT/run.spec['reuse_molecule_cache_from']
        frozen=json.loads((source/'manifest.json').read_text())
        assert frozen['encoder_manifest_sha256']==sha(run.encoder/'manifest.json')
        assert sha(source/'molecules.parquet')==sha(run.out/'molecules.parquet'),'Molecular row order changed'
        marker=json.loads((source/'molecules_encoded.json').read_text())
        values=np.load(source/'molecule_embeddings.npy',mmap_mode='r')
        assert values.shape==(marker['rows'],768) and values.dtype==np.float16
        for start in range(0,len(values),8192):assert np.isfinite(values[start:start+8192]).all()
        digest=sha(source/'molecule_embeddings.npy');target=run.out/'molecule_embeddings.npy'
        if target.exists():assert sha(target)==digest
        else:os.link(source/'molecule_embeddings.npy',target)
        save_json(run.out/'molecule_cache_reuse.json',dict(source=str(source),sha256=digest,
                  source_manifest_sha256=sha(source/'manifest.json'),identical_molecules_and_encoder=True))
        save_json(run.out/'molecules_encoded.json',marker)
        return
    from transformers import AutoModel,AutoTokenizer
    run.status('loading frozen MoLFormer')
    folder=run.encoder/'molformer'
    tokenizer=AutoTokenizer.from_pretrained(folder,trust_remote_code=True,local_files_only=True)
    model=AutoModel.from_pretrained(folder,trust_remote_code=True,local_files_only=True,deterministic_eval=True).cuda().eval()
    model.requires_grad_(False)
    mol=pd.read_parquet(run.out/'molecules.parquet')
    @torch.inference_mode()
    def encode(rows):
        inputs=tokenizer(mol.smiles.iloc[rows].tolist(),padding=True,truncation=True,max_length=202,return_tensors='pt')
        result=model(**{k:v.cuda() for k,v in inputs.items()}).pooler_output.float().cpu().numpy()
        return result
    encode_shards(run,'molecule_embeddings',np.arange(len(mol)),768,encode)
    save_json(run.out/'molecules_encoded.json',dict(rows=len(mol),model=run.encoder_manifest['molecule_model'],revision=run.encoder_manifest['molecule_revision'],pooling='pooler_output',storage='float16'))
    del model;gc.collect();torch.cuda.empty_cache()


def spectrum_cache(run,pilot):
    marker=run.out/('pilot_spectra_encoded.json' if pilot else 'all_spectra_encoded.json')
    if marker.exists():return
    os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'  # Only checksum-verified original author checkpoints.
    import dreams.api as api
    from dreams.models.heads.heads import ContrastiveHead
    api.PRETRAINED=run.encoder/'dreams'
    run.status('loading frozen DreaMS')
    model=api.PreTrainedModel.from_ckpt(run.encoder/'dreams/embedding_model.ckpt',ContrastiveHead,100).model.cuda().eval()
    model.requires_grad_(False)
    peaks=np.load(run.out/'spectra.npy',mmap_mode='r')
    selected=pd.read_parquet(run.out/'selected.parquet');split=np.load(run.out/'split.npz',allow_pickle=False)
    pilot_mol=np.r_[split['pilot'],split['development'],split['confirmation']]
    mask=selected.mol_idx.isin(pilot_mol).to_numpy()
    ids=np.flatnonzero(mask&selected.valid.to_numpy()) if pilot else np.flatnonzero((~mask)&selected.valid.to_numpy())
    @torch.inference_mode()
    def encode(rows):return model(torch.tensor(np.asarray(peaks[rows]),device='cuda')).float().cpu().numpy()
    encode_shards(run,'spectrum_embeddings',ids,1024,encode)
    save_json(marker,dict(encoded_rows=len(ids),supported_queries_only=True))
    del model;gc.collect();torch.cuda.empty_cache()


def make_candidates(run):
    path=run.out/'candidates.npz'
    if path.exists():return
    run.status('mining training-only mass candidates')
    mol=pd.read_parquet(run.out/'molecules.parquet');split=np.load(run.out/'split.npz',allow_pickle=False)
    training=split['train'];order=training[np.argsort(mol.mass.to_numpy()[training],kind='stable')]
    masses=mol.mass.to_numpy();sorted_mass=masses[order]
    count=run.spec['training']['candidates']
    negatives=np.full((len(mol),count),-1,dtype=np.int32)
    roots=np.load(ROOT/run.spec['split_groups'],allow_pickle=False)['groups']
    for index,i in enumerate(training):
        pos=int(np.searchsorted(sorted_mass,masses[i]));near=order[max(0,pos-count*4):pos+count*4+1]
        near=near[(near!=i)&(roots[near]!=roots[i])];near=near[np.argsort(np.abs(masses[near]-masses[i]),kind='stable')]
        assert len(near)>=count
        negatives[i]=near[:count]
        if index%10000==0:run.status('mining training-only mass candidates',stage_completed=index,stage_total=len(training))
    train_mask=np.zeros(len(mol),dtype=bool);train_mask[training]=True
    assert train_mask[negatives[training]].all()
    evaluation=np.r_[split['development'],split['confirmation']]
    all_order=np.argsort(masses,kind='stable');all_mass=masses[all_order];candidates=[]
    for i in evaluation:
        pos=int(np.searchsorted(all_mass,masses[i]));near=all_order[max(0,pos-256):pos+257]
        near=near[near!=i];near=near[np.argsort(np.abs(masses[near]-masses[i]),kind='stable')]
        candidates.append(np.r_[i,near[:255]])
    np.savez(path,training_negatives=negatives,evaluation_ids=evaluation,evaluation_candidates=np.array(candidates,np.int32))


def load_arrays(run):
    selected=pd.read_parquet(run.out/'selected.parquet');n=len(pd.read_parquet(run.out/'molecules.parquet'))
    active=np.flatnonzero(selected.valid.to_numpy());row_mol=selected.mol_idx.to_numpy()[active]
    order=active[np.argsort(row_mol,kind='stable')];offset=np.r_[0,np.bincount(row_mol,minlength=n).cumsum()]
    return dict(rows=order,offset=offset,selected=selected,
                spectrum=np.load(run.out/'spectrum_embeddings.npy',mmap_mode='r'),
                molecule=np.load(run.out/'molecule_embeddings.npy',mmap_mode='r'),
                metadata=np.load(run.out/'metadata.npy',mmap_mode='r'),adduct=np.load(run.out/'adduct.npy',mmap_mode='r'))


@torch.inference_mode()
def evaluate(model,data,candidates):
    model.eval();scores=[]
    for i,choices in zip(candidates['evaluation_ids'],candidates['evaluation_candidates']):
        rows=data['rows'][data['offset'][i]:data['offset'][i+1]]
        z=model.encode_spectra(torch.tensor(np.asarray(data['spectrum'][rows]),device='cuda',dtype=torch.float32),
            torch.tensor(np.asarray(data['metadata'][rows]),device='cuda'),torch.tensor(np.asarray(data['adduct'][rows]),device='cuda')).mean(0)
        m=model.encode_molecules(torch.tensor(np.asarray(data['molecule'][choices]),device='cuda',dtype=torch.float32))
        scores.append((m@z/model.log_temperature.exp().clamp(.01,1)).cpu().numpy())
    scores=np.asarray(scores);ranks=1+(scores[:,1:]>=scores[:,:1]).sum(1)
    return np.where(ranks<=25,1/ranks,0.),scores


def train(run,seed,pilot=False):
    cfg=run.spec['training'];folder=run.out/'models'/f'seed-{seed}';folder.mkdir(parents=True,exist_ok=True)
    target=run.spec['pilot_steps'] if pilot else cfg['steps']
    checkpoint=folder/'resume.pt'
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=ProjectionRanker(cfg['hidden'],cfg['dimension'],cfg['dropout']).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=1e-6)
    scaler=torch.amp.GradScaler('cuda');rng=np.random.default_rng(seed);step=0;history=[];best=-1.
    if checkpoint.exists():
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        model.load_state_dict(state['state_dict']);optimizer.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler'])
        step=state['step'];history=state['history'];best=state['best_development']
        rng.bit_generator.state=state['numpy_rng'];random.setstate(state['python_rng']);torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
    if step>=target:return
    data=load_arrays(run);split=np.load(run.out/'split.npz',allow_pickle=False);candidate=np.load(run.out/'candidates.npz',allow_pickle=False)
    ids=split['pilot'] if pilot else split['train'];negatives=candidate['training_negatives'];batch=cfg['batch_size']
    training_mask=np.zeros(len(data['offset'])-1,dtype=bool);training_mask[split['train']]=True
    n_dev=len(split['development'])
    dev_candidate=dict(evaluation_ids=candidate['evaluation_ids'][:n_dev],evaluation_candidates=candidate['evaluation_candidates'][:n_dev])
    # Entire 5% fold is excluded, including positives, negatives and learned auxiliary features.
    assert (split['folds'][ids]!=0).all()
    initial_mrr=None
    if pilot and step==0:
        rr,_=evaluate(model,data,dev_candidate);initial_mrr=float(rr.mean())
        save_json(folder/'initial.json',dict(development_mrr25=initial_mrr))
    elif (folder/'initial.json').exists(): initial_mrr=json.loads((folder/'initial.json').read_text())['development_mrr25']
    losses=[];started=time.monotonic();last=time.monotonic()
    for step in range(step+1,target+1):
        model.train();chosen=rng.choice(ids,batch,replace=False)
        rows=data['rows'][data['offset'][chosen]+(rng.random(batch)*(data['offset'][chosen+1]-data['offset'][chosen])).astype(np.int64)]
        choices=np.column_stack([chosen,negatives[chosen]])
        assert training_mask[choices].all()
        optimizer.zero_grad(set_to_none=True)
        progress=max(0,(step-cfg['warmup_steps'])/max(1,cfg['steps']-cfg['warmup_steps']))
        lr=cfg['lr']*min(1,step/cfg['warmup_steps'])*(.05+.95*.5*(1+math.cos(math.pi*progress)))
        for group in optimizer.param_groups:group['lr']=lr
        with torch.autocast('cuda',dtype=torch.float16):
            scores=model(torch.tensor(np.asarray(data['spectrum'][rows]),device='cuda',dtype=torch.float32),
                torch.tensor(np.asarray(data['metadata'][rows]),device='cuda'),torch.tensor(np.asarray(data['adduct'][rows]),device='cuda'),
                torch.tensor(np.asarray(data['molecule'][choices]),device='cuda',dtype=torch.float32))
            loss=F.cross_entropy(scores.float(),torch.zeros(batch,device='cuda',dtype=torch.long))
        assert torch.isfinite(loss),'Nonfinite head loss'
        scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),5)
        scaler.step(optimizer);scaler.update();losses.append(float(loss))
        if step%100==0:
            run.status('pilot head training' if pilot else 'full head training',seed=seed,step=step,steps=cfg['steps'],loss=float(np.mean(losses[-100:])))
        if step%cfg['evaluation_interval']==0 or step==target:
            rr,scores=evaluate(model,data,dev_candidate);dev=float(rr.mean())
            record=dict(step=step,loss=float(np.mean(losses)),development_mrr25=dev,seconds=time.monotonic()-last,pilot=pilot)
            history.append(record);losses=[];last=time.monotonic()
            if dev>best:
                best=dev;save_torch(folder/'best.pt',dict(state_dict=model.state_dict(),step=step,seed=seed,cfg=cfg,development_mrr25=dev))
            save_torch(checkpoint,dict(state_dict=model.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
                step=step,history=history,best_development=best,numpy_rng=rng.bit_generator.state,python_rng=random.getstate(),
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()))
            save_json(folder/'history.json',history);print('HEAD',seed,json.dumps(record),flush=True)
    if pilot:
        assert initial_mrr is not None and best>initial_mrr and all(torch.isfinite(p).all() for p in model.parameters()),'Pilot did not improve over untrained heads'
        save_json(run.out/'pilot_passed.json',dict(seed=seed,steps=target,initial_development_mrr25=initial_mrr,development_mrr25=best,
                   meaning='Execution/learning pilot only; no production or leaderboard qualification'))
    else:
        best_state=torch.load(folder/'best.pt',map_location='cuda',weights_only=True);model.load_state_dict(best_state['state_dict'])
        rr,scores=evaluate(model,data,candidate);n=len(split['development'])
        np.savez(folder/'evaluation.npz',ids=candidate['evaluation_ids'],scores=scores,rr=rr)
        save_json(folder/'complete.json',dict(seed=seed,steps=target,best_step=best_state['step'],
            development_mrr25=float(rr[:n].mean()),confirmation_mrr25=float(rr[n:].mean()),model_sha256=sha(folder/'best.pt'),
            limitation='Truth-forced mass-near retrieval; production end-to-end confirmation still required'))
    del model,optimizer,scaler;gc.collect();torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,default=ROOT/'configs/frozen-training-20261107.json')
    args=parser.parse_args();spec=json.loads(args.plan.read_text())
    assert not (ROOT/'.local/compute-paused.json').exists(),'Project paused'
    torch.set_num_threads(2);torch.set_num_interop_threads(1);pa.set_cpu_count(2)
    assert torch.cuda.is_available() and '2070' in torch.cuda.get_device_name(0)
    torch.cuda.set_per_process_memory_fraction(spec['gpu_memory_fraction'],0)
    run=Run(spec)
    try:
        prepare(run);molecule_cache(run);make_candidates(run);spectrum_cache(run,True)
        train(run,spec['seeds'][0],True)
        spectrum_cache(run,False)
        for seed in spec['seeds']:train(run,seed,False)
        results=[json.loads((run.out/'models'/f'seed-{seed}'/'complete.json').read_text()) for seed in spec['seeds']]
        save_json(run.out/'results.json',dict(models=results,automatic_submission=False))
        run.status('complete');state=json.loads((run.out/'status.json').read_text());state['state']='complete';save_json(run.out/'status.json',state)
        print('DONE FROZEN RETRIEVAL',json.dumps(results),flush=True)
    except BaseException as error:
        save_json(run.out/'failure.json',dict(error=repr(error),utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid()))
        raise


if __name__=='__main__':main()
