import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F



class GaussianLoss(nn.Module):
    def __init__(self, N, sigma=0.3, device='cpu', *args, **kwargs):
        super(GaussianLoss, self).__init__()

        self.sigma = torch.full((N, N), sigma, device=device)

    def forward(self, pred, target):
        distr = torch.distributions.Normal(target, self.sigma)

        bound_1 = distr.cdf(pred)
        bound_2 = distr.cdf(2 * target - pred)
        log = 1.0000000001 - torch.abs(bound_1 - bound_2)
        
        log_prob = torch.log(log)

        loss = -torch.mean(log_prob)

        return loss
        

class LogL1Loss(nn.Module):
    def __init__(self, *args, **kwargs):
        super(LogL1Loss, self).__init__()

    def forward(self, pred, target):
        l1_loss = 1 - F.l1_loss(pred, target) / 2

        log_l1_loss = -torch.log(l1_loss)

        return log_l1_loss

    def numpy_loss(self, pred, target):
        l1_loss = 1 - np.mean(np.abs(pred - target)) / 2

        log_l1_loss = -np.log(l1_loss)

        return log_l1_loss


class AbsMeanDeltaLoss(nn.Module): # NOTE: it supposes that the probabilistic distribution around the value is symetric
    def __init__(self, *args, **kwargs):
        super(AbsMeanDeltaLoss, self).__init__()

    def forward(self, pred, target):
        mean_delta = torch.mean(pred - target)
        return torch.abs(mean_delta)
        

class L1LossAdjust(nn.Module):
    def __init__(self, adjust, device, *args, **kwargs):
        super(L1LossAdjust, self).__init__()
        self.adjust = torch.tensor(adjust, device=device)

    def forward(self, pred, target):
        l1_loss = F.l1_loss(pred, target)

        return torch.abs(l1_loss - self.adjust)


class MinLoss(nn.Module):
    def __init__(self):
        super(MinLoss, self).__init__()

    def forward(self, pred, target):
        l1_loss = torch.abs(pred - target).min(axis=0)
