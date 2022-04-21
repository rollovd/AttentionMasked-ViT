import torch
import torch.nn as nn
from torch import Tensor


class WMVLoss(nn.Module):

    def __init__(self, p, gamma):
        super(WMVLoss, self).__init__()
        self.p = p
        self.gamma = gamma

    def forward(self, inputs: Tensor, target: Tensor) -> Tensor:
        if len(inputs.shape) > 1:
            scores = torch.softmax(inputs, dim=-1)[:, 1]
        else:
            scores = inputs
        positive_idx = target == 1
        negative_idx = torch.logical_not(positive_idx)
        diff = scores[None, positive_idx] - scores[negative_idx, None]
        selected_diff = diff[diff < self.gamma]
        rs = torch.pow(-(selected_diff - self.gamma), self.p)
        return torch.sum(rs) / inputs.size(0)


