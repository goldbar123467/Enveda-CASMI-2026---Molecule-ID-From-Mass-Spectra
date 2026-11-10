"""Collision-energy masking with a separate, checkpointable RNG."""
import torch


def mask_known_ce(batch,probability,generator):
    assert 0 <= probability <= 1
    metadata=batch[2]
    assert metadata.ndim==2 and metadata.shape[1]==4 and torch.isfinite(metadata).all()
    assert ((metadata[:,2]==0)|(metadata[:,2]==1)).all()
    eligible=metadata[:,2]>0
    selected=(torch.rand(len(metadata),device=metadata.device,generator=generator)<probability)&eligible
    after=metadata.clone();after[selected,1:3]=0
    return (batch[0],batch[1],after,batch[3]),dict(eligible=int(eligible.sum().item()),
        masked=int(selected.sum().item()),unknown_before=int((~eligible).sum().item()))
