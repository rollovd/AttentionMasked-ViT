import torch
from torch.utils.data.sampler import WeightedRandomSampler

def ClassImbalanceSampler(labels):
    class_count = torch.bincount(labels.squeeze())
    class_weighting = 1. / class_count
    sample_weights = class_weighting[labels]
    sampler = WeightedRandomSampler(sample_weights, len(labels))
    return sampler