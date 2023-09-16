import torch
from torch import nn


EPS = 1e-10


class AsymmetricBCELoss(nn.Module):
    def __init__(self, scale=2.0):
        super().__init__()
        self.scale = scale

    def forward(self, predict, target):
        predict = predict.view((-1, 1))
        target = target.view((-1, 1))
        loss = -self.scale * target * torch.log(predict + EPS) - (1 - target) * torch.log(1 - predict + EPS)
        return torch.nanmean(loss)


class AsymmetricMSELoss(nn.Module):
    def __init__(self, scale=2.0):
        super().__init__()
        self.scale = scale

    def forward(self, predict, target):
        predict = predict.view((-1, 1))
        target = target.view((-1, 1))
        mask = (predict - target) >= 0
        part1 = (predict[mask] - target[mask]) ** 2 if mask.any() else torch.Tensor([0])
        part2 = (predict[~mask] - target[~mask]) ** 2 if ~mask.all() else torch.Tensor([0])
        return self.scale * torch.nanmean(part1) + torch.nanmean(part2)
