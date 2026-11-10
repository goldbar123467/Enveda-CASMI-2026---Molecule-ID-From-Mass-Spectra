"""Drop/compact valid peaks using a separate checkpointable augmentation RNG."""
import torch


def drop_peaks(batch,probability,generator):
    assert 0<=probability<=1
    peaks,lengths,metadata,adduct=batch
    assert peaks.ndim==3 and peaks.shape[2]==4 and torch.isfinite(peaks).all()
    n,width,_=peaks.shape
    assert lengths.shape==(n,) and ((lengths>=3)&(lengths<=width)).all()
    positions=torch.arange(width,device=peaks.device).expand(n,-1)
    valid=positions<lengths[:,None]
    intensity=peaks[:,:,1].masked_fill(~valid,-torch.inf)
    protected=torch.argsort(intensity,dim=1,descending=True,stable=True)[:,:3]
    keep=(torch.rand((n,width),device=peaks.device,generator=generator)>=probability)&valid
    keep.scatter_(1,protected,True)
    after_lengths=keep.sum(1).to(lengths.dtype)
    order=torch.argsort(positions+(~keep)*width,dim=1,stable=True)
    after=peaks.gather(1,order[:,:,None].expand(-1,-1,4))
    after=after.masked_fill((positions>=after_lengths[:,None])[:,:,None],0)
    assert (after_lengths>=3).all() and (after_lengths<=lengths).all()
    return (after,after_lengths,metadata,adduct),dict(peaks_before=int(lengths.sum().item()),
        peaks_after=int(after_lengths.sum().item()),spectra=int(n),spectra_changed=int((after_lengths<lengths).sum().item()))
