"""A bounded spectral residual; the structure-fingerprint geometry stays fixed."""
import torch
from torch import nn
from torch.nn import functional as F


class BoundedResidualHead(nn.Module):
    def __init__(self,cfg,anchor_weights=None):
        super().__init__();self.cfg=cfg
        if anchor_weights is None:anchor_weights=torch.zeros(1040,2048)
        assert tuple(anchor_weights.shape)==(1040,2048)
        self.register_buffer('anchor_weights',anchor_weights.detach().clone().float())
        h=cfg['hidden']
        self.residual=nn.Sequential(nn.Linear(1039,h),nn.LayerNorm(h),nn.GELU(),nn.Dropout(cfg['dropout']),nn.Linear(h,2048))
        nn.init.zeros_(self.residual[-1].weight);nn.init.zeros_(self.residual[-1].bias)

    def components(self,x):
        anchor=x@self.anchor_weights[:-1]+self.anchor_weights[-1]
        raw=self.residual(x)
        delta=self.cfg['maximum_relative_residual_norm']*anchor.norm(dim=-1,keepdim=True)*raw/(1+raw.norm(dim=-1,keepdim=True))
        return anchor,delta

    def encode_spectra(self,x):
        anchor,delta=self.components(x);return F.normalize(anchor+delta,dim=-1)

    def encode_molecules(self,y):return F.normalize(y,dim=-1)
