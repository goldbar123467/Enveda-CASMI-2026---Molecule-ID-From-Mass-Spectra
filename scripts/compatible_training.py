"""Shared AMP objective for the compatible-scorer experiment (credited Saner kit)."""
import torch
from torch.nn import functional as F


def accumulated_step(model,optimizer,scaler,batch,positive,negative,micro_batch,cfg,duplicate_mask=None,pos_weight=None):
    """Retain the full logical batch's positive competitors with small query batches.

    Molecule embeddings are recomputed per microbatch; this avoids retaining a
    shared autograd graph. Dropout draws may differ; this is not a bit-exact
    replacement for a monolithic training step. Both experiment arms use it.
    """
    n=len(positive);optimizer.zero_grad(set_to_none=True);total=0.
    for start in range(0,n,micro_batch):
        stop=min(n,start+micro_batch)
        with torch.autocast('cuda',dtype=torch.float16):
            z,p=model.encode_spectra(*(value[start:stop] for value in batch))
            m=model.encode_molecules(positive)
            nm=model.encode_molecules(negative[start:stop].flatten(0,1)).reshape(stop-start,negative.shape[1],-1)
            hard=torch.einsum('bd,bkd->bk',z,nm)/cfg['temperature']
            if duplicate_mask is not None:hard=hard.masked_fill(duplicate_mask[start:stop],float('-inf'))
            logits=torch.cat([z@m.T/cfg['temperature'],hard],1)
            targets=torch.arange(start,stop,device=z.device)
            loss=F.cross_entropy(logits.float(),targets)
            loss=loss+cfg['aux_weight']*F.binary_cross_entropy_with_logits(p.float(),positive[start:stop],pos_weight=pos_weight)
            loss=loss*((stop-start)/n)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite compatible training loss')
        scaler.scale(loss).backward();total+=float(loss.detach())
    scaler.unscale_(optimizer);norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    old_scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
    return dict(loss=total,gradient_norm=float(norm),amp_skipped=scaler.get_scale()<old_scale)
