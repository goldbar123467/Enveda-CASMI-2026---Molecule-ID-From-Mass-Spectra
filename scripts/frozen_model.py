"""Small trainable retrieval heads over immutable public encoder features."""
import torch
from torch import nn
from torch.nn import functional as F


class ProjectionRanker(nn.Module):
    def __init__(self, hidden=1024, dimension=512, dropout=.2):
        super().__init__()
        self.adduct = nn.Embedding(11, 32)
        self.metadata = nn.Sequential(nn.Linear(4, 32), nn.GELU())
        def mlp(n):
            return nn.Sequential(nn.Linear(n, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, dimension))
        self.spectrum = mlp(1024 + 64)
        self.molecule = mlp(768)
        self.log_temperature = nn.Parameter(torch.tensor(.07).log())

    def encode_spectra(self, embeddings, metadata, adduct):
        return F.normalize(self.spectrum(torch.cat([embeddings, self.metadata(metadata), self.adduct(adduct)], -1)), dim=-1)

    def encode_molecules(self, embeddings):
        return F.normalize(self.molecule(embeddings), dim=-1)

    def forward(self, spectra, metadata, adduct, molecules):
        query = self.encode_spectra(spectra, metadata, adduct)
        candidate = self.encode_molecules(molecules)
        return torch.einsum('bd,bkd->bk', query, candidate) / self.log_temperature.exp().clamp(.01, 1)
