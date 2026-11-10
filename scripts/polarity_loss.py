"""Importance-weight query losses toward equal positive/negative exposure."""
import torch
from torch.nn import functional as F


def query_weights(polarity,negative_probability):
    assert 0<negative_probability<1 and ((polarity==1)|(polarity==-1)).all()
    return torch.where(polarity<0,.5/negative_probability,.5/(1-negative_probability))


def weighted_loss(logits,targets,fp_logits,fp_targets,pos_weight,polarity,negative_probability,aux_weight,logical_batch_size):
    ce=F.cross_entropy(logits.float(),targets,reduction='none')
    bce=F.binary_cross_entropy_with_logits(fp_logits.float(),fp_targets,pos_weight=pos_weight,reduction='none').mean(-1)
    weights=query_weights(polarity,negative_probability)
    assert ce.shape==bce.shape==weights.shape and logical_batch_size>=len(ce)
    return ((ce+aux_weight*bce)*weights).sum()/logical_batch_size
