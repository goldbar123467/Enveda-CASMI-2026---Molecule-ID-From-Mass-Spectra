"""Sequential, resumable RTX2070 shared-fold experiment; never submits to Kaggle.

Inspired by cdeotte/kaggle-playground-solutions' common folds, independent model
folders and OOF prediction contracts. This is molecular retrieval, so identities
replace tabular row folds and candidate scores replace binary probabilities.
"""
import argparse
import atexit
from collections import defaultdict
import ctypes
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import struct
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as NN

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT/'outputs/local/kit'
sys.path.insert(0, str(KIT))
from model import SpectrumCandidateModel
import train as T


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def json_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temp.replace(path)


def refresh_ledger(out):
    try:
        result=subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),str(ROOT/'scripts/update_runs.py')],
                              cwd=ROOT,capture_output=True,text=True,timeout=120)
        with (out/'ledger_refresh.log').open('a',encoding='utf-8') as f:
            f.write(result.stdout+result.stderr)
    except Exception as exc:
        print('LEDGER_REFRESH_WARNING',repr(exc),flush=True)


def finish_ledger(run_id, status, result):
    path=ROOT/'runs.csv'
    with path.open(newline='',encoding='utf-8') as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
    found=[row for row in rows if row['id']==run_id]
    assert len(found)==1, 'Training controller must be registered before launch'
    found[0].update(status=status,result=result,ended_utc=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M'))
    temp=path.with_suffix('.csv.tmp')
    with temp.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    temp.replace(path)


def fold_for(key, folds=5, salt='casmi-20261106-v1'):
    return int.from_bytes(hashlib.sha256((salt+':'+str(key)).encode()).digest()[:8], 'little') % folds


def training_ids(fold_ids, fold, base_count, external):
    ids = np.flatnonzero(np.asarray(fold_ids) != fold)
    return ids if external else ids[ids < base_count]


def metrics(scores):
    ranks = 1+(scores[:,1:] >= scores[:,:1]).sum(1)
    return dict(n=len(ranks), mrr25=float(np.where(ranks<=25,1/ranks,0).mean()),
                top1=float((ranks==1).mean()), recall25=float((ranks<=25).mean()))


class RowsByMolecule:
    def __init__(self, ids, n, row_ids=None):
        order = np.argsort(ids, kind='stable')
        self.order = order if row_ids is None else np.asarray(row_ids)[order]
        self.offsets = np.r_[0, np.bincount(ids, minlength=n).cumsum()]
    def __getitem__(self, i):
        return self.order[self.offsets[i]:self.offsets[i+1]]


class Data:
    """Memory-map peaks; hold only small indexes and uint8 fingerprints in RAM."""
    def __init__(self, base, external, cfg, split_cache):
        self.cfg = cfg
        bm = pd.read_parquet(base/'molecules.parquet')
        em = pd.read_parquet(external/'molecules.parquet')
        assert bm.key.is_unique and em.key.is_unique
        self.base_count = len(bm)
        lookup = dict(zip(bm.key, range(len(bm))))
        new = em[~em.key.isin(lookup)].copy()
        self.mol = pd.concat([bm[['key','smiles','formula','mass']], new], ignore_index=True)
        lookup = dict(zip(self.mol.key, range(len(self.mol))))
        external_ids = em.key.map(lookup).to_numpy(np.int64)
        assert len(self.mol) == len(set(self.mol.key))
        bfp = np.load(base/'fingerprints.npy', mmap_mode='r')
        efp = np.load(external/'fingerprints.npy', mmap_mode='r')
        # Verify overlapping labels use the SAME Morgan fingerprint and bit order.
        overlap = np.flatnonzero(external_ids < self.base_count)
        compatible=np.ones(len(em),dtype=bool)
        compatible[overlap]=np.all(np.asarray(bfp[external_ids[overlap]])==np.asarray(efp[overlap]),axis=1)
        mismatched=np.flatnonzero(~compatible)
        assert len(mismatched) < max(1,.01*len(overlap)), 'Widespread fingerprint mismatch; inspect featurization'
        # E.g. S(=O) vs [S+]([O-]) can share the metric identity but give different
        # Morgan vectors. Exclude these ambiguous external labels rather than
        # silently replacing the existing structure encoder's training targets.
        json_save(split_cache/'excluded_external_labels.json',dict(count=len(mismatched),
                  keys=em.key.iloc[mismatched].tolist(),reason='Existing metric identity has a different fingerprint representation'))
        self.fp = np.concatenate([bfp, efp[external_ids >= self.base_count]], axis=0)
        bs = pd.read_parquet(base/'spectra.parquet', columns=['mol_idx','length','adduct_id'])
        es = pd.read_parquet(external/'spectra.parquet', columns=['mol_idx','length','adduct_id','source'])
        eligible_external=compatible[es.mol_idx.to_numpy()]
        es['mol_idx'] = external_ids[es.mol_idx.to_numpy()]
        self.base_spectra = len(bs)
        self.lengths = np.r_[bs.length.to_numpy(), es.length.to_numpy()]
        self.adduct = np.r_[bs.adduct_id.to_numpy(), es.adduct_id.to_numpy()]
        self.row_molecule = np.r_[bs.mol_idx.to_numpy(), es.mol_idx.to_numpy()]
        self.training_rows=np.r_[np.ones(len(bs),dtype=bool),eligible_external]
        self.base_rows = RowsByMolecule(bs.mol_idx.to_numpy(), len(self.mol))
        self.x = [np.memmap(p/'peaks.bin', mode='r', dtype='<f2', shape=(n,cfg['max_peaks'],4))
                  for p,n in [(base,len(bs)),(external,len(es))]]
        self.meta = [np.memmap(p/'metadata.bin', mode='r', dtype='<f4', shape=(n,4))
                     for p,n in [(base,len(bs)),(external,len(es))]]
        self.native = np.load(ROOT/'outputs/local/paired-native/native_rows.npy')
        assert self.native.max() < self.base_spectra
        self.native_rows = RowsByMolecule(self.row_molecule[self.native], len(self.mol))
        self.folds = spectrum_group_folds(self,base,external,split_cache)
        eligible=np.flatnonzero(self.training_rows)
        self.by_mol=RowsByMolecule(self.row_molecule[eligible],len(self.mol),eligible)
        print('DATA', json.dumps(dict(molecules=len(self.mol),base_molecules=len(bm),external_new_molecules=len(new),
                                     external_overlap=len(overlap),base_spectra=len(bs),external_spectra=len(es),
                                     excluded_fingerprint_identities=len(mismatched),
                                     external_eligible_spectra=int(self.training_rows[self.base_spectra:].sum()))), flush=True)

    def query_rows(self, i):
        native = self.native[self.native_rows[i]]
        return native if len(native) else self.base_rows[i]

    def batch(self, rows, device, augment=False):
        rows = np.asarray(rows)
        x = np.empty((len(rows),self.cfg['max_peaks'],4), np.float32)
        meta = np.empty((len(rows),4), np.float32)
        for j in (0,1):
            mask = rows < self.base_spectra if j==0 else rows >= self.base_spectra
            local = rows[mask] if j==0 else rows[mask]-self.base_spectra
            x[mask] = self.x[j][local]
            meta[mask] = self.meta[j][local]
        x = torch.as_tensor(x, device=device)
        if augment:
            it = (x[:,:,1].square()*torch.exp(.08*torch.randn_like(x[:,:,1]))).clamp_min(0)
            it /= it.max(1,keepdim=True).values.clamp_min(1e-8)
            x[:,:,1] = it.sqrt()
            x[:,:,2] = torch.log1p(100*it)/np.log(101)
        return (x, torch.as_tensor(self.lengths[rows],device=device,dtype=torch.long),
                torch.as_tensor(meta,device=device),torch.as_tensor(self.adduct[rows],device=device,dtype=torch.long))


def merge_groups(parent, a, b):
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=int(parent[i])
        return i
    a,b=find(int(a)),find(int(b))
    if a!=b:parent[max(a,b)]=min(a,b)


def spectrum_group_folds(data,base,external,out):
    """Tie labels connected by identical model inputs, even if labels disagree.

    Cache quantization can create extra equivalences; grouping them is deliberately
    conservative. It prevents a different-label copy of a query becoming training
    evidence. Identity-only folds would miss this known data-quality problem.
    """
    path=out/'spectrum_groups.npz';audit_path=out/'spectrum_groups.json'
    inputs={str(p):sha(p) for root in [base,external] for p in
            [root/'peaks.bin',root/'metadata.bin',root/'spectra.parquet',root/'molecules.parquet']}
    if path.exists():
        audit=json.loads(audit_path.read_text())
        assert audit['inputs']==inputs and audit['artifact_sha256']==sha(path)
        saved=np.load(path,allow_pickle=False)
        assert not np.any(saved['training_rows'] & ~data.training_rows)
        data.training_rows=saved['training_rows']
        return saved['folds']
    parent=np.arange(len(data.mol),dtype=np.int64)
    seen={};links=0;row_offset=0;external_duplicates=0
    for source in range(2):
        for row in range(len(data.x[source])):
            global_row=row_offset+row
            if not data.training_rows[global_row]:continue
            signature=hashlib.blake2b(data.x[source][row].tobytes()+data.meta[source][row].tobytes()+
                      struct.pack('<HB',int(data.lengths[global_row]),int(data.adduct[global_row])),digest_size=16).digest()
            mol=int(data.row_molecule[global_row])
            old=seen.get(signature)
            if old is None:seen[signature]=mol
            elif old!=mol:
                merge_groups(parent,old,mol);links+=1
            elif source==1:
                data.training_rows[global_row]=False;external_duplicates+=1
            if row and row%500000==0:print('SPLIT_AUDIT',source,row,'cross_label_input_matches',links,flush=True)
        row_offset+=len(data.x[source])
    for i in range(len(parent)):
        while parent[i]!=parent[parent[i]]:parent[i]=parent[parent[i]]
    keys=data.mol.key.to_numpy()
    folds=np.array([fold_for(keys[root]) for root in parent],dtype=np.int8)
    np.savez_compressed(path,groups=parent,folds=folds,training_rows=data.training_rows)
    json_save(audit_path,dict(inputs=inputs,molecules=len(parent),groups=len(np.unique(parent)),
              cross_label_input_matches=links,duplicate_external_inputs_removed=external_duplicates,
              external_eligible_spectra=int(data.training_rows[data.base_spectra:].sum()),
              unique_input_signatures=len(seen),artifact_sha256=sha(path),
              rule='Connected components of byte-identical quantized model inputs share a fold; identity duplicates share an index'))
    return folds


def evaluation_plan(data, out, per_fold):
    path = out/'evaluation.npz'
    np_keys = set((ROOT/'outputs/local/seeds/np_holdout_keys.txt').read_text().split())
    chosen = []
    for fold in range(5):
        ids = np.flatnonzero(data.folds[:data.base_count] == fold)
        ordered = sorted(ids, key=lambda i:hashlib.sha256(('evaluation:'+data.mol.key.iloc[i]).encode()).digest())
        # All test-like public NP examples, plus a fixed fresh sample per fold.
        selected = set(ordered[:per_fold]) | {int(i) for i in ids if data.mol.key.iloc[i] in np_keys}
        chosen.extend(sorted(selected))
    chosen = np.asarray(chosen, np.int64)
    # Common candidates from the original structure universe, independent of condition.
    frame = data.mol.iloc[:data.base_count]
    groups = defaultdict(list)
    for i,f in enumerate(frame.formula):
        groups[f].append(i)
    mass_order = np.argsort(frame.mass.to_numpy(),kind='stable')
    masses = frame.mass.to_numpy()[mass_order]
    candidates = []
    for i in chosen:
        pos = int(np.searchsorted(masses,frame.mass.iloc[i]))
        near = mass_order[max(0,pos-256):pos+257]
        near = sorted(near,key=lambda j:(abs(frame.mass.iloc[j]-frame.mass.iloc[i]),int(j)))
        decoys = list(dict.fromkeys([i]+groups[frame.formula.iloc[i]]+near))[:256]
        assert len(decoys)==256 and decoys[0]==i
        candidates.append(decoys)
    result = dict(ids=chosen,keys=data.mol.key.iloc[chosen].to_numpy(dtype=str),
                  folds=data.folds[chosen],candidates=np.asarray(candidates,np.int64),
                  natural_product=np.array([data.mol.key.iloc[i] in np_keys for i in chosen]))
    if path.exists():
        prior = np.load(path,allow_pickle=False)
        assert all(np.array_equal(prior[k],v) for k,v in result.items()), 'Evaluation plan changed on resume'
    else:
        np.savez_compressed(path,**result)
    return result


@torch.inference_mode()
def evaluate(model, data, plan, fold, device, cfg):
    model.eval()
    selected = np.flatnonzero(plan['folds']==fold)
    cosine, fingerprint = [], []
    for j in selected:
        i = int(plan['ids'][j])
        assert data.folds[i] == fold
        rows = data.query_rows(i)
        zsum, psum = 0, 0
        for start in range(0,len(rows),cfg['batch_size']):
            z,p = model.encode_spectra(*data.batch(rows[start:start+cfg['batch_size']],device))
            zsum += z.sum(0)
            psum += p.sum(0)
        fp = torch.tensor(data.fp[plan['candidates'][j]],device=device,dtype=torch.float32)
        cosine.append((model.encode_molecules(fp)@(zsum/len(rows))/cfg['temperature']).cpu().numpy())
        fingerprint.append((cfg['fp_score_weight']*(fp@(psum/len(rows)))/cfg['fp_bits']).cpu().numpy())
    cos, fps = np.asarray(cosine), np.asarray(fingerprint)
    return dict(keys=plan['keys'][selected],ids=plan['ids'][selected],folds=plan['folds'][selected],
                candidates=plan['candidates'][selected],natural_product=plan['natural_product'][selected],
                cosine=cos,fingerprint=fps,scores=cos+fps)


def free_ram_gib():
    if os.name != 'nt':
        return 999.
    class Memory(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(n,ctypes.c_ulonglong) for n in
                  ['totalPhysical','availablePhysical','totalPage','availablePage','totalVirtual','availableVirtual','availableExtended']]
    mem=Memory();mem.length=ctypes.sizeof(mem)
    assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))
    return mem.availablePhysical/2**30


def fit(data, plan, cfg, out, fold, external, smoke=False, deadline=None, heartbeat=None):
    family = 'external' if external else 'control'
    folder = out/'models'/family/f'fold-{fold}'
    folder.mkdir(parents=True,exist_ok=True)
    pred = folder/'preds/oof.npz'
    final = folder/'model.pt'
    contract = dict(fold=fold,family=family,cfg=cfg,smoke=smoke,evaluation_sha256=sha(out/'evaluation.npz'),
                    external_manifest_sha256=sha(Path(cfg['external_cache'])/'manifest.json'),
                    trainer_sha256=sha(__file__),base_cache_sha256=sha(ROOT/'outputs/local/n0/cache/molecules.parquet'))
    manifest = folder/'manifest.json'
    if manifest.exists():
        assert json.loads(manifest.read_text()) == contract, 'Run contract changed; use a new output directory'
    else:
        json_save(manifest,contract)
    if final.exists() and pred.exists() and (folder/'complete.json').exists():
        receipt=json.loads((folder/'complete.json').read_text())
        assert receipt['model_sha256']==sha(final) and receipt['predictions_sha256']==sha(pred)
        print('SKIP verified complete',family,fold,flush=True)
        return
    ids = training_ids(data.folds,fold,data.base_count,external)
    if smoke:
        ids=np.r_[ids[:896],ids[-128:]] if external else ids[:1024]
        ids=np.unique(ids)
    assert not np.any(data.folds[ids]==fold)
    seed=cfg['seed']+fold
    T.seed_all(seed)
    rng=np.random.default_rng(seed)
    device=torch.device('cuda')
    model=SpectrumCandidateModel(cfg).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    scaler=torch.amp.GradScaler('cuda')
    history=[]
    resume=folder/'resume.pt'
    if resume.exists():
        saved=torch.load(resume,map_location='cpu',weights_only=True)
        assert saved['contract']==contract
        model.load_state_dict(saved['state_dict']);optimizer.load_state_dict(saved['optimizer'])
        scaler.load_state_dict(saved['scaler']);history=saved['history']
        rng.bit_generator.state=saved['numpy_rng']
        random.setstate(saved['python_rng']);torch.set_rng_state(saved['torch_rng'])
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
    if smoke:
        allowed=ids[:256]
        pools={int(i):allowed[allowed!=i][:cfg['negative_pool']] for i in ids}
    else:
        pools=T.hard_pools(data.mol,ids,cfg['negative_pool'])
    counts=np.zeros(cfg['fp_bits'],np.float64)
    for start in range(0,len(ids),4096):
        counts += data.fp[ids[start:start+4096]].sum(0)
    weights=torch.tensor(np.clip((len(ids)-counts)/(counts+1),1,20),device=device,dtype=torch.float32)
    json_save(folder/'split_audit.json',dict(training=len(ids),heldout=int((data.folds==fold).sum()),
              training_key_sha256=hashlib.sha256('\n'.join(sorted(data.mol.key.iloc[ids])).encode()).hexdigest(),
              excluded_fold=fold,external_overlap_excluded_across_all_sources=True))
    for epoch in range(len(history)+1,cfg['epochs']+1):
        if deadline is not None and time.monotonic() > deadline:
            raise RuntimeError('Session budget reached at epoch boundary; resume from saved state')
        if free_ram_gib() < cfg['minimum_free_ram_gib']:
            raise RuntimeError('RAM reserve reached; resume from the last complete epoch')
        started=time.monotonic();model.train();losses=[];skipped=0;external_rows=0
        torch.cuda.reset_peak_memory_stats()
        for step in range(cfg['steps_per_epoch']):
            fraction=((epoch-1)*cfg['steps_per_epoch']+step)/(cfg['epochs']*cfg['steps_per_epoch'])
            lr=cfg['lr']*(.05+.95*.5*(1+math.cos(math.pi*fraction)))
            for group in optimizer.param_groups:group['lr']=lr
            batch_ids=rng.choice(ids,cfg['batch_size'],replace=False)
            source_rows=data.by_mol if external else data.base_rows
            rows=np.array([rng.choice(source_rows[int(i)]) for i in batch_ids])
            external_rows+=int((rows>=getattr(data,'base_spectra',len(data.row_molecule))).sum())
            assert np.all(data.folds[data.row_molecule[rows]]!=fold)
            neg=np.stack([rng.choice(pools[int(i)],cfg['hard_negatives'],replace=False) for i in batch_ids])
            assert np.all(data.folds[neg]!=fold)
            fp=torch.tensor(data.fp[batch_ids],device=device,dtype=torch.float32)
            nf=torch.tensor(data.fp[neg.reshape(-1)],device=device,dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):
                z,p=model.encode_spectra(*data.batch(rows,device,augment=True))
                m=model.encode_molecules(fp)
                nm=model.encode_molecules(nf).reshape(len(batch_ids),cfg['hard_negatives'],-1)
                positive=z@m.T/cfg['temperature']
                negative=torch.einsum('bd,bkd->bk',z,nm)/cfg['temperature']
                duplicate=torch.tensor(np.isin(neg,batch_ids),device=device)
                negative=negative.masked_fill(duplicate,float('-inf'))
                loss=NN.cross_entropy(torch.cat([positive,negative],1),torch.arange(len(rows),device=device))
                loss=loss+cfg['aux_weight']*NN.binary_cross_entropy_with_logits(p,fp,pos_weight=weights)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
            skipped+=int(scaler.get_scale()<scale)
            losses.append(float(loss.detach()))
        record=dict(epoch=epoch,steps=len(losses),loss=float(np.mean(losses)),seconds=round(time.monotonic()-started,2),
                    lr=lr,amp_skipped_steps=skipped,sampled_external_spectra=external_rows,
                    gpu_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                    gpu_peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,free_ram_gib=free_ram_gib())
        history.append(record)
        state=dict(contract=contract,state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
                   optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),history=history,numpy_rng=rng.bit_generator.state,
                   python_rng=random.getstate(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
        T.atomic_save(state,resume)
        del state
        json_save(folder/'history.json',history)
        json_save(out/'status.json',dict(state='training',family=family,fold=fold,epoch=epoch,epochs=cfg['epochs'],
                  pid=os.getpid(),updated_utc=datetime.now(timezone.utc).isoformat(),resources=record))
        print('EPOCH',family,fold,json.dumps(record),flush=True)
        if heartbeat is not None:heartbeat()
    assert len(history)==cfg['epochs'] and all(h['steps']==cfg['steps_per_epoch'] for h in history)
    # Final epoch is fixed in advance; no best-epoch selection or holdout-driven early stopping.
    assert all(torch.isfinite(v).all() for v in model.state_dict().values())
    assert sum(h['steps']-h['amp_skipped_steps'] for h in history)>0, 'No successful optimizer step'
    if external and smoke:assert sum(h['sampled_external_spectra'] for h in history)>0, 'External smoke never sampled external data'
    prediction=evaluate(model,data,plan,fold,device,cfg)
    pred.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(pred,**prediction)
    T.atomic_save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},cfg=cfg,
                      stage='crossfit_holdout',epoch=cfg['epochs'],fold=fold,smoke=smoke,contract=contract),final)
    result=metrics(prediction['scores'])
    json_save(folder/'complete.json',dict(epochs=cfg['epochs'],steps=sum(h['steps'] for h in history),
              metrics=result,model_sha256=sha(final),predictions_sha256=sha(pred),smoke=smoke))
    # Only the active job needs optimizer state; final inference weights and OOF stay.
    resume.unlink()
    print('FOLD_COMPLETE',family,fold,json.dumps(result),flush=True)
    del optimizer,model,scaler,pools
    torch.cuda.empty_cache()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--plan',type=Path,default=ROOT/'configs/long-training-20261106.json')
    ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args()
    if (ROOT/'.local/compute-paused.json').exists():raise RuntimeError('Project paused')
    spec=json.loads(args.plan.read_text())
    out=ROOT/spec['output']
    if args.smoke:out=out.parent/(out.name+'-smoke')
    out.mkdir(parents=True,exist_ok=True)
    # Prevent accidental simultaneous copies of this controller on Windows.
    lock=(out/'controller.lock').open('a+b')
    if os.name=='nt':
        import msvcrt
        lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        # Process-scoped system-awake request, restored on exit; no power-plan edit.
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        atexit.register(lambda:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000))
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    assert torch.cuda.is_available() and '2070' in torch.cuda.get_device_name(0)
    torch.cuda.set_per_process_memory_fraction(spec['gpu_memory_fraction'],0)
    cfg=dict(json.loads((KIT/'config.json').read_text()),**spec['training'])
    cfg.update(external_cache=str(ROOT/spec['external_cache']),minimum_free_ram_gib=spec['minimum_free_ram_gib'])
    if args.smoke:cfg.update(epochs=1,steps_per_epoch=8)
    data=Data(ROOT/'outputs/local/n0/cache',Path(cfg['external_cache']),cfg,out)
    plan=evaluation_plan(data,out,4 if args.smoke else spec['evaluation_molecules_per_fold'])
    frozen=dict(spec=spec,cfg=cfg,smoke=args.smoke,fold_algorithm='SHA256(salt:canonical key) modulo5; connected identical model-input label groups stay together',
                spectrum_group_sha256=sha(out/'spectrum_groups.npz'),
                fingerprint_sha256={str(p):sha(p) for p in [ROOT/'outputs/local/n0/cache/fingerprints.npy',Path(cfg['external_cache'])/'fingerprints.npy']},
                evaluation_sha256=sha(out/'evaluation.npz'),no_test_file_used=True,baseline_public_mrr=.480,
                limitation='Truth-forced retrieval OOF is diagnostic; does not measure full candidate recall or Kaggle MRR',
                source_sha256={p:sha(ROOT/p) for p in ['scripts/long_training.py','scripts/stack_oof.py',
                    'scripts/build_external_cache.py','outputs/local/kit/train.py','outputs/local/kit/model.py','outputs/local/kit/features.py']})
    if (out/'manifest.json').exists():assert json.loads((out/'manifest.json').read_text())==frozen
    else:json_save(out/'manifest.json',frozen)
    for relative,expected_hash in frozen['source_sha256'].items():
        destination=out/'sources'/relative
        if destination.exists():assert sha(destination)==expected_hash
        else:
            destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_bytes((ROOT/relative).read_bytes())
    started=time.monotonic()
    refreshed=[started]
    def heartbeat():
        if time.monotonic()-refreshed[0] >= 3600:
            refresh_ledger(out);refreshed[0]=time.monotonic()
    try:
        for fold in range(1 if args.smoke else 5):
            for external in (False,True):
                assert all(sha(ROOT/p)==h for p,h in frozen['source_sha256'].items()), 'Source changed during training'
                if time.monotonic()-started > spec['maximum_hours']*3600:raise RuntimeError('Session budget reached; resume remaining folds')
                fit(data,plan,cfg,out,fold,external,args.smoke,started+spec['maximum_hours']*3600,heartbeat)
        if not args.smoke:
            assert all(sha(ROOT/p)==h for p,h in frozen['source_sha256'].items()), 'Source changed before stacking'
            subprocess.run([sys.executable,str(ROOT/'scripts/stack_oof.py'),str(out)],check=True,cwd=ROOT)
        json_save(out/'status.json',dict(state='complete',smoke=args.smoke,pid=os.getpid(),
                  updated_utc=datetime.now(timezone.utc).isoformat(),next='OOF stack finished; qualify on full-pipeline replay before any submission'))
        print('[0s] DONE all shared-fold trainings; smoke='+str(args.smoke),flush=True)
        if not args.smoke:
            report=json.loads((out/'stack/report.json').read_text())['cohorts']['confirmation']
            summary='10 models x128x500 complete; '+', '.join(f"{key} confirmation retrieval MRR={report[key]['mrr25']:.6f}" for key in ['control','external','equal_blend','l2_stack'])+'; diagnostic only, no submission.'
            finish_ledger(spec['run_id'],'complete',summary)
            refresh_ledger(out)
    except Exception as exc:
        json_save(out/'failure.json',dict(error=repr(exc),pid=os.getpid(),utc=datetime.now(timezone.utc).isoformat()))
        if not args.smoke:
            finish_ledger(spec['run_id'],'error','Incomplete shared-fold run; resumable epoch states retained. '+repr(exc))
            refresh_ledger(out)
        raise


if __name__=='__main__':
    main()
