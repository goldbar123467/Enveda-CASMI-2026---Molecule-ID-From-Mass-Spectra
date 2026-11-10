"""Same-identity/polarity two-view averaging, matching the serving arithmetic."""
import copy,math
import numpy as np
import torch
from torch.nn import functional as F


def select_mates(data,ids,first_rows,rng,probability=.5):
    assert 0<=probability<=1 and len(ids)==len(first_rows)
    draws=rng.random((len(ids),2));second=np.array(first_rows,copy=True);use=np.zeros(len(ids),bool)
    for i,(identity,first) in enumerate(zip(ids,first_rows)):
        rows=data.order[data.offset[identity]:data.offset[identity+1]]
        assert first in rows and data.meta[first,3] in [-1,1]
        choices=rows[(data.meta[rows,3]==data.meta[first,3])&(rows!=first)]
        if len(choices) and draws[i,0]<probability:
            second[i]=choices[min(int(draws[i,1]*len(choices)),len(choices)-1)];use[i]=True
    return second,use


def average_encoded(z,p,extra_z,extra_p,indices):
    """Do not renormalize averaged embeddings: serving averages raw fold scores."""
    if len(indices)==0:return z,p
    return z.index_copy(0,indices,(z[indices]+extra_z)*.5),p.index_copy(0,indices,(p[indices]+extra_p)*.5)


class ViewSampler:
    def __init__(self,data,seed,probability=.5):
        self.data=data;self.rng=np.random.default_rng(seed);self.probability=probability
        self.jitter_rng=torch.Generator(device='cuda').manual_seed(seed+1)
        self.counts=dict(primary_queries=0,two_view_queries=0,single_view_queries=0)

    def state_dict(self):return dict(rng=self.rng.bit_generator.state,jitter_rng=self.jitter_rng.get_state(),counts=self.counts)

    def load_state_dict(self,state):
        self.rng.bit_generator.state=state['rng'];self.jitter_rng.set_state(state['jitter_rng']);self.counts=dict(state['counts'])

    def batch(self,ids,negative,query_rng,precision):
        probe=np.random.default_rng();probe.bit_generator.state=copy.deepcopy(query_rng.bit_generator.state)
        rows=self.data.order[self.data.offset[ids]+(probe.random(len(ids))*np.diff(self.data.offset)[ids]).astype(np.int64)]
        first,pos,neg,mask=self.data.batch(ids,negative,query_rng,precision)
        second,use=select_mates(self.data,ids,rows,self.rng,self.probability)
        assert np.array_equal(self.data.meta[rows,3],self.data.meta[second,3])
        peaks=np.array(self.data.x[second],dtype=np.float32)
        if precision=='float16':peaks=peaks.astype(np.float16).astype(np.float32)
        x=torch.as_tensor(peaks,device='cuda')
        jitter=torch.randn(x[:,:,1].shape,device=x.device,dtype=x.dtype,generator=self.jitter_rng)
        intensity=(x[:,:,1].square()*torch.exp(.08*jitter)).clamp_min(0)
        intensity/=intensity.max(1,keepdim=True).values.clamp_min(1e-8)
        x[:,:,1]=intensity.sqrt();x[:,:,2]=torch.log1p(100*intensity)/math.log(101)
        extra=(x,torch.as_tensor(np.array(self.data.lengths[second]),device='cuda',dtype=torch.long),
            torch.as_tensor(np.array(self.data.meta[second]),device='cuda'),torch.as_tensor(np.array(self.data.adduct[second]),device='cuda',dtype=torch.long))
        self.counts['primary_queries']+=len(ids);self.counts['two_view_queries']+=int(use.sum());self.counts['single_view_queries']+=int((~use).sum())
        return (first,extra,use),pos,neg,mask


def update(model,optimizer,scaler,batch,pos,neg,mask,micro,cfg,pos_weight):
    first,second,use=batch;n=len(pos);optimizer.zero_grad(set_to_none=True);total=0.
    for start in range(0,n,micro):
        stop=min(start+micro,n);indices=torch.as_tensor(np.flatnonzero(use[start:stop]),device=pos.device,dtype=torch.long)
        with torch.autocast('cuda',dtype=torch.float16):
            z,p=model.encode_spectra(*(x[start:stop] for x in first))
            if len(indices):
                zz,pp=model.encode_spectra(*(x[start:stop][indices] for x in second))
                z,p=average_encoded(z,p,zz,pp,indices)
            molecule=model.encode_molecules(pos)
            negative=model.encode_molecules(neg[start:stop].flatten(0,1)).reshape(stop-start,neg.shape[1],-1)
            primary=z@molecule.T/cfg['temperature'];hard=torch.einsum('bd,bkd->bk',z,negative)/cfg['temperature']
            hard=hard.masked_fill(mask[start:stop],float('-inf'))
            loss=F.cross_entropy(torch.cat([primary,hard],1).float(),torch.arange(start,stop,device=pos.device))
            loss+=cfg['aux_weight']*F.binary_cross_entropy_with_logits(p.float(),pos[start:stop],pos_weight=pos_weight)
            loss*=float(stop-start)/n
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite C28 multiview loss')
        scaler.scale(loss).backward();total+=float(loss.detach())
    scaler.unscale_(optimizer);norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    old=scaler.get_scale();scaler.step(optimizer);scaler.update()
    return dict(loss=total,gradient_norm=float(norm),skipped=scaler.get_scale()<old)
