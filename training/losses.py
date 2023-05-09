import torch
import torch.nn as nn
import torch.nn.functional as F



class GaussianLoss(nn.Module):
    def __init__(self, N, sigma=0.3, device='cpu'):
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
    def __init__(self, N, sigma=0.3, device='cpu'):
        super(LogL1Loss, self).__init__()

    def forward(self, pred, target):
        l1_loss = 1 - F.l1_loss(pred, target) / 2

        log_l1_loss = -torch.mean(torch.log(l1_loss))

        return log_l1_loss

