"""Deterministic online hard-negative selection without gradient/RNG mutation."""
import torch


def hardest_indices(scores,mask,keep):
    assert scores.ndim==2 and scores.shape==mask.shape and 0<keep<=scores.shape[1]
    assert torch.isfinite(scores).all()
    masked=scores.float().masked_fill(mask,float('-inf'))
    # Ties retain source-pool order across checkpoint recovery.
    return torch.argsort(masked,dim=1,descending=True,stable=True)[:,:keep]


@torch.no_grad()
def mine(model,batch,positive,pool,mask,cfg,keep,encode_chunk=2048):
    was_training=model.training
    try:
        model.eval()
        with torch.autocast(device_type=pool.device.type,dtype=torch.float16,enabled=pool.is_cuda):
            z,p=model.encode_spectra(*batch)
            flat=pool.flatten(0,1)
            vectors=torch.cat([model.encode_molecules(flat[start:start+encode_chunk]) for start in range(0,len(flat),encode_chunk)])
            vectors=vectors.reshape(len(pool),pool.shape[1],-1)
            scores=torch.einsum('bd,bkd->bk',z,vectors)/cfg['temperature']
            scores=scores+cfg['fp_score_weight']/cfg['fp_bits']*torch.einsum('bd,bkd->bk',p,pool)
        # The serving architecture cannot distinguish equal Morgan vectors.
        # Mask such impossible negatives, as well as all logical-batch positives.
        equivalent=(pool==positive[:,None,:]).all(-1)
        combined=mask|equivalent
        order=hardest_indices(scores,combined,keep)
        selected=pool.gather(1,order[:,:,None].expand(-1,-1,pool.shape[-1])).contiguous()
        selected_mask=combined.gather(1,order)
        values=scores.float().masked_fill(combined,0.)
        pool_mean=values.sum(1)/(~combined).sum(1).clamp_min(1)
        chosen=scores.float().gather(1,order).masked_fill(selected_mask,0.)
        chosen_mean=chosen.sum(1)/(~selected_mask).sum(1).clamp_min(1)
        gain=chosen_mean-pool_mean
        assert torch.isfinite(gain).all() and float(gain.min())>=-1e-4
        stats=dict(mining_gain=float(gain.mean()),equivalent_fingerprints_masked=int(equivalent.sum()),
            selected_valid_min=int((~selected_mask).sum(1).min()),pool_candidates=pool.shape[1],hard_negatives=keep)
        return selected,selected_mask,stats
    finally:model.train(was_training)
