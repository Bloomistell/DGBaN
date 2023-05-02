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

        x = torch.sigmoid(x).squeeze(dim=1) * 1.4

        return x, kl_sum



class big_DGBaNR(torch.nn.Module): # R for reparametrization
    def __init__(self, input_size, img_size, activation_function, pre_trained_base=False):
        super(big_DGBaNR, self).__init__()

        self.img_size = img_size

        self.linear1 = BayesLinearR(input_size, 256)
        self.linear2 = BayesLinearR(256, 512 * 4 * 4)

        self.conv1 = BayesConvT2dR(512, 256)
        self.batch_norm1 = nn.BatchNorm2d(256)

        self.conv2 = BayesConvT2dR(256, 128)
        self.batch_norm2 = nn.BatchNorm2d(128)

        self.conv3 = BayesConvT2dR(128, 1)

        if activation_function == 'sigmoid':
            self.activation_function = torch.sigmoid
        else:
            self.activation_function = getattr(F, activation_function)

    def forward(self, x):
        kl_sum = 0

        x, kl = self.linear1(x)
        kl_sum += kl
        x = F.relu(x)
        
        x, kl = self.linear2(x)
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

        x = self.activation_function(x).squeeze(dim=1)

        return x, kl_sum

        
# b is for bayesian and the number stands for how many bayesian layers the model has
class multi_b1conv_DGBaNR(torch.nn.Module): # the idea for this one is to keep the conv part deterministic, because the image pattern as wholes are submitted to probabilistic appearences
    def __init__(self, input_size, img_size, activation_function):
        super(multi_b1conv_DGBaNR, self).__init__()

        self.img_size = img_size

        self.linear_1 = nn.Linear(input_size // 2, 54)
        self.linear_2 = nn.Linear(input_size // 2, 54)

        self.linear_layers = nn.Sequential(
            nn.ReLU(),
            nn.Linear(108, 972),
            nn.ReLU(),
            nn.Linear(972, 512 * 4 * 4),
            nn.ReLU()
        )

        self.conv_layers = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.bayes_conv = BayesConvT2dR(128, 1)

        if activation_function == 'sigmoid':
            self.activation_function = torch.sigmoid
        else:
            self.activation_function = getattr(F, activation_function)

    def forward(self, x):
        x_1, x_2 = torch.split(x, 6, dim=1)
        x_1 = self.linear_1(x_1)
        x_2 = self.linear_2(x_2)

        x = torch.cat((x_1, x_2), dim=1)
        x = self.linear_layers(x)
        
        x = x.view(x.size(0), 512, 4, 4)
        x = self.conv_layers(x)

        x, kl = self.bayes_conv(x)

        x = self.activation_function(x).squeeze(dim=1)

        return x, kl



class b1conv_DGBaNR(torch.nn.Module): # the idea for this one is to keep the conv part deterministic, because the image pattern as wholes are submitted to probabilistic appearences
    def __init__(self, input_size, img_size, activation_function):
        super(b1conv_DGBaNR, self).__init__()

        self.img_size = img_size

        self.linear_layers = nn.Sequential(
            nn.Linear(input_size, 108),
            nn.ReLU(),
            nn.Linear(108, 972),
            nn.ReLU(),
            nn.Linear(972, 512 * 4 * 4),
            nn.ReLU()
        )

        self.conv_layers = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.bayes_conv = BayesConvT2dR(128, 1)

        if activation_function == 'sigmoid':
            self.activation_function = torch.sigmoid
        else:
            self.activation_function = getattr(F, activation_function)

    def forward(self, x):
        x = self.linear_layers(x)
        
        x = x.view(x.size(0), 512, 4, 4)
        x = self.conv_layers(x)

        x, kl = self.bayes_conv(x)

        x = self.activation_function(x).squeeze(dim=1)

        return x, kl



class multi_bconv_DGBaNR(torch.nn.Module): # the idea for this one is to keep the conv part deterministic, because the image pattern as wholes are submitted to probabilistic appearences
    def __init__(self, input_size, img_size, activation_function):
        super(multi_bconv_DGBaNR, self).__init__()

        self.img_size = img_size

        self.linear_1 = nn.Linear(input_size // 2, 54)
        self.linear_2 = nn.Linear(input_size // 2, 54)

        self.linear_layers = nn.Sequential(
            nn.ReLU(),
            nn.Linear(108, 972),
            nn.ReLU(),
            nn.Linear(972, 512 * 4 * 4),
            nn.ReLU()
        )

        self.conv1 = BayesConvT2dR(512, 256)
        self.batch_norm1 = nn.BatchNorm2d(256)

        self.conv2 = BayesConvT2dR(256, 128)
        self.batch_norm2 = nn.BatchNorm2d(128)

        self.conv3 = BayesConvT2dR(128, 1)

        if activation_function == 'sigmoid':
            self.activation_function = torch.sigmoid
        else:
            self.activation_function = getattr(F, activation_function)

    def forward(self, x):
        x_1, x_2 = torch.split(x, 6, dim=1)
        x_1 = self.linear_1(x_1)
        x_2 = self.linear_2(x_2)

        x = torch.cat((x_1, x_2), dim=1)
        x = self.linear_layers(x)

        kl_sum = 0
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

        x = self.activation_function(x).squeeze(dim=1)

        return x, kl_sum



class bconv_DGBaNR(torch.nn.Module): # the idea for this one is to keep the conv part deterministic, because the image pattern as wholes are submitted to probabilistic appearences
    def __init__(self, input_size, img_size, activation_function):
        super(bconv_DGBaNR, self).__init__()

        self.img_size = img_size

        self.linear_layers = nn.Sequential(
            nn.Linear(input_size, 108),
            nn.ReLU(),
            nn.Linear(108, 972),
            nn.ReLU(),
            nn.Linear(972, 512 * 4 * 4),
            nn.ReLU()
        )

        self.conv1 = BayesConvT2dR(512, 256)
        self.batch_norm1 = nn.BatchNorm2d(256)

        self.conv2 = BayesConvT2dR(256, 128)
        self.batch_norm2 = nn.BatchNorm2d(128)

        self.conv3 = BayesConvT2dR(128, 1)

        if activation_function == 'sigmoid':
            self.activation_function = torch.sigmoid
        else:
            self.activation_function = getattr(F, activation_function)

    def forward(self, x):
        x = self.linear_layers(x)

        kl_sum = 0
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

        x = self.activation_function(x).squeeze(dim=1)

        return x, kl_sum



class bbuffer_DGBaNR(torch.nn.Module): 
    def __init__(self, input_size, img_size, activation_function, pre_trained_base=False):
        super(bbuffer_DGBaNR, self).__init__()

        self.img_size = img_size

        self.linear_layers = nn.Sequential(
            nn.Linear(input_size, 108),
            nn.ReLU(),
            nn.Linear(108, 972),
            nn.ReLU(),
            nn.Linear(972, 512 * 4 * 4),
            nn.ReLU()
        )

        self.conv_layers = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1)
        )

        if pre_trained_base:
            for param in self.linear_layers.parameters():
                param.requires_grad = False

            for param in self.conv_layers.parameters():
                param.requires_grad = False

        self.bayes_1 = LinearReparameterization(
            1024,
            1024,
            prior_mean=0,
            prior_variance=1,
            posterior_mu_init=0,
            posterior_rho_init=-0.3
            )
        self.batch_norm_1 = nn.BatchNorm1d(1024)
        self.bayes_2 = LinearReparameterization(
            1024,
            1024,
            prior_mean=0,
            prior_variance=1,
            posterior_mu_init=0,
            posterior_rho_init=-0.3
        )
        self.batch_norm_2 = nn.BatchNorm1d(1024)

        if activation_function == 'sigmoid':
            self.activation_function = torch.sigmoid
        else:
            self.activation_function = getattr(F, activation_function)

    def forward(self, x):
        x = self.linear_layers(x)

        x = x.view(x.size(0), 512, 4, 4)
        x = self.conv_layers(x)
        
        kl_sum = 0
        x = x.flatten(start_dim=1)
        x, kl = self.bayes_1(x)
        kl_sum += kl
        x = self.batch_norm_1(x)
        x, kl = self.bayes_2(x)
        kl_sum += kl
        x = self.batch_norm_2(x)

        x = self.activation_function(x)

        return x.reshape(x.size(0), 32, 32), kl_sum