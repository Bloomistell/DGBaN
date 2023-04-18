import torch
import torch.nn as nn
import torch.nn.functional as F

from bayesian_torch.layers.variational_layers.linear_variational import LinearReparameterization
from bayesian_torch.layers.variational_layers.conv_variational import ConvTranspose2dReparameterization



prior_mu = 0.0
prior_sigma = 1.0
posterior_mu_init = 0.0
posterior_rho_init = -3.0


def BayesLinearR(n_input, n_output, stride=2):
    return LinearReparameterization(
        n_input,
        n_output,
        prior_mean=prior_mu,
        prior_variance=prior_sigma,
        posterior_mu_init=posterior_mu_init,
        posterior_rho_init=posterior_rho_init
    )

def BayesConvT2dR(in_planes, out_planes, stride=2):
    return ConvTranspose2dReparameterization(
        in_channels=in_planes,
        out_channels=out_planes,
        kernel_size=4,
        stride=stride,
        padding=1,
        prior_mean=prior_mu,
        prior_variance=prior_sigma,
        posterior_mu_init=posterior_mu_init,
        posterior_rho_init=posterior_rho_init
    )


class DGBaNR(torch.nn.Module): # R for reparametrization
    def __init__(self, input_size, img_size):
        super(DGBaNR, self).__init__()

        self.img_size = img_size

        self.linear = BayesLinearR(input_size, 512 * 4 * 4)

        self.conv1 = BayesConvT2dR(512, 256)
        self.batch_norm1 = nn.BatchNorm2d(256)

        self.conv2 = BayesConvT2dR(256, 128)
        self.batch_norm2 = nn.BatchNorm2d(128)

        self.conv3 = BayesConvT2dR(128, 1)

    def forward(self, x):
        kl_sum = 0

        x, kl = self.linear(x)
        kl_sum += kl
        x = F.relu(x)
        
        x = x.view(x.size(0), 512, 4, 4)

        x, kl = self.conv1(x)
        kl_sum += kl
        x = self.batch_norm1(x)
        x = F.relu(x)

        x, kl = self.conv2(x)
        kl_sum += kl
        x = self.batch_norm2(x)
        x = F.relu(x)
        
        x, kl = self.conv3(x)
        kl_sum += kl

        x = torch.sigmoid(x).squeeze(dim=1)

        return x, kl_sum

