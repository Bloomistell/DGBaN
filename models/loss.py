import torch
import torch.nn.functional as F



# class ELBO(nn.Module):
#     def __init__(self, train_size):
#         super(ELBO, self).__init__()
#         self.train_size = train_size

#     def forward(self, input, target, kl, beta):
#         assert not target.requires_grad
#         return F.nll_loss(input, target, reduction='mean') * self.train_size + beta * kl