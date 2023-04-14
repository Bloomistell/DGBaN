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
    ConvTranspose2dReparametrization as BayesConvT2dR
)



class LinearGenerator(torch.nn.Module):
    def __init__(self, input_size, img_size):
        super(LinearGenerator, self).__init__()

        self.img_size = img_size

        self.network = Sequential(
            Linear(input_size, 12),
            ReLU(),

            Linear(12, 36),
            BatchNorm1d(36),
            ReLU(),

            Linear(36, 108),
            BatchNorm1d(108),
            ReLU(),

            Linear(108, 324),
            BatchNorm1d(324),
            ReLU(),

            Linear(324, img_size * img_size),
            Sigmoid()
        )

    def forward(self, x):
        x = self.network(x)
        return x.view(x.size(0), self.img_size, self.img_size)



class ConvGenerator(torch.nn.Module):
    def __init__(self, input_size, img_size):
        super(ConvGenerator, self).__init__()

        self.img_size = img_size

        self.neural_net = Sequential(
            Linear(input_size, 512 * 4 * 4),
            ReLU(),
        )

        self.conv_net = Sequential(
            ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            BatchNorm2d(256),
            ReLU(),
            
            ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            BatchNorm2d(128),
            ReLU(),

            ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1),
            Sigmoid()
        )

    def forward(self, x):
        x = self.neural_net(x)
        img = self.conv_net(x.view(x.size(0), 512, 4, 4)).squeeze()
        return img



prior_mu = 0.0
prior_sigma = 1.0
posterior_mu_init = 0.0
posterior_rho_init = -3.0

class DGBaNR(torch.nn.Module): # R for reparametrization
    def __init__(self, input_size, img_size):
        super(ConvGenerator, self).__init__()

        self.img_size = img_size

        self.neural_net = Sequential(
            BayesLinearR(
            input_size,
            512 * 4 * 4,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
        )
            ReLU(),
        )

        self.conv_net = Sequential(
            BayesConvT2dR(
                512,
                256,
                kernel_size=4,
                stride=2,
                prior_mean=prior_mu,
                prior_variance=prior_sigma,
                posterior_mu_init=posterior_mu_init,
                posterior_rho_init=posterior_rho_init
            ),
            BatchNorm2d(256),
            ReLU(),
            
            BayesConvT2dR(
                256,
                128,
                kernel_size=4,
                stride=2,
                prior_mean=prior_mu,
                prior_variance=prior_sigma,
                posterior_mu_init=posterior_mu_init,
                posterior_rho_init=posterior_rho_init
            ),
            BatchNorm2d(128),
            ReLU(),

            BayesConvT2dR(
                128,
                1,
                kernel_size=4,
                stride=2,
                prior_mean=prior_mu,
                prior_variance=prior_sigma,
                posterior_mu_init=posterior_mu_init,
                posterior_rho_init=posterior_rho_init
            ),
            Sigmoid()
        )

    def forward(self, x):
        x = self.neural_net(x)
        img = self.conv_net(x.view(x.size(0), 512, 4, 4)).squeeze()
        return img




