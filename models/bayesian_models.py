import torch
from torch.nn import (
    Linear,
    Sequential,
    ReLU,
    BatchNorm1d,
    Sigmoid,
    ConvTranspose2d,
    BatchNorm2d
)

from bayesian_torch.layers import (
    LinearReparametrization as BayesLinearR,
    ConvTranspose2dReparametrization
)



prior_mu = 0.0
prior_sigma = 1.0
posterior_mu_init = 0.0
posterior_rho_init = -3.0

def BayesConvT2dR(in_planes, out_planes, stride=2):
    return Conv2dReparameterization(in_channels=in_planes,
                                    out_channels=out_planes,
                                    kernel_size=4,
                                    stride=stride,
                                    padding=1,
                                    prior_mean=prior_mu,
                                    prior_variance=prior_sigma,
                                    posterior_mu_init=posterior_mu_init,
                                    posterior_rho_init=posterior_rho_init)


class DGBaNR(torch.nn.Module): # R for reparametrization
    def __init__(self, input_size, img_size):
        super(ConvGenerator, self).__init__()

        self.img_size = img_size

        self.neural_net = Sequential(
            BayesLinearR(input_size, 512 * 4 * 4)
            ReLU(),
        )

        self.conv_net = Sequential(
            BayesConvT2dR(512, 256),
            BatchNorm2d(256),
            ReLU(),
            
            BayesConvT2dR(256, 128),
            BatchNorm2d(128),
            ReLU(),

            BayesConvT2dR(128, 1),
            Sigmoid()
        )

    def forward(self, x):
        x = self.neural_net(x)
        img = self.conv_net(x.view(x.size(0), 512, 4, 4)).squeeze()
        return img




