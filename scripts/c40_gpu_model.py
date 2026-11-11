"""Small nonlinear head over immutable encoded features."""
import torch
from torch import nn
from torch.nn import functional as F


class CrossModalHead(nn.Module):
    def __init__(self,cfg):
        super().__init__();h=cfg['hidden'];d=cfg['dimension']
        def mlp(n):return nn.Sequential(nn.Linear(n,h),nn.LayerNorm(h),nn.GELU(),nn.Dropout(cfg['dropout']),nn.Linear(h,d))
        self.spectrum=mlp(1039);self.molecule=mlp(768)
        self.log_temperature=nn.Parameter(torch.tensor(float(cfg['temperature'])).log())
    def encode_spectra(self,x):return F.normalize(self.spectrum(x),dim=-1)
    def encode_molecules(self,y):return F.normalize(self.molecule(y),dim=-1)
    def forward(self,x,y):
        return self.encode_spectra(x)@self.encode_molecules(y).T/self.log_temperature.exp().clamp(.01,1.)
