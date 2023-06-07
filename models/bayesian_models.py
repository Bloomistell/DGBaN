import torch
import torch.nn as nn
import torch.nn.functional as F

from bayesian_torch.layers.variational_layers.linear_variational import LinearReparameterization as BayesLinear
from bayesian_torch.layers.variational_layers.conv_variational import (
    ConvTranspose2dReparameterization as BayesConvTranspose2d,
    Conv2dReparameterization as BayesConv2d
)

import models.bayesian_blocks as blocks



prior_mu = 0.0
prior_sigma = 1.0
posterior_mu_init = 0.0
posterior_rho_init = -3.0


def BayesLinearR(n_input, n_output):
    return BayesLinear(
        n_input,
        n_output,
        prior_mean=prior_mu,
        prior_variance=prior_sigma,
        posterior_mu_init=posterior_mu_init,
        posterior_rho_init=posterior_rho_init
    )

def BayesConvT2dR(in_planes, out_planes, stride=2):
    return BayesConvTranspose2d(
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

        self.base_name = 'big_DGBaNR_base'

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
        self.bayes_3 = LinearReparameterization(
            1024,
            1024,
            prior_mean=0,
            prior_variance=1,
            posterior_mu_init=0,
            posterior_rho_init=-0.3
        )
        self.batch_norm_3 = nn.BatchNorm1d(1024)
        self.bayes_4 = LinearReparameterization(
            1024,
            1024,
            prior_mean=0,
            prior_variance=1,
            posterior_mu_init=0,
            posterior_rho_init=-0.3
        )
        self.batch_norm_4 = nn.BatchNorm1d(1024)

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
        x, kl = self.bayes_3(x)
        kl_sum += kl
        x = self.batch_norm_3(x)
        x, kl = self.bayes_4(x)
        kl_sum += kl
        x = self.batch_norm_4(x)

        x = self.activation_function(x)

        return x.reshape(x.size(0), 32, 32), kl_sum
    
    

class recall_bbuffer_DGBaNR(torch.nn.Module): 
    def __init__(self, input_size, img_size, activation_function, pre_trained_base=False):
        super(recall_bbuffer_DGBaNR, self).__init__()

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
            input_size + 1024,
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
        self.bayes_3 = LinearReparameterization(
            1024,
            1024,
            prior_mean=0,
            prior_variance=1,
            posterior_mu_init=0,
            posterior_rho_init=-0.3
        )
        self.batch_norm_3 = nn.BatchNorm1d(1024)

        if activation_function == 'sigmoid':
            self.activation_function = torch.sigmoid
        else:
            self.activation_function = getattr(F, activation_function)

    def forward(self, x):
        z = x.clone()
        x = self.linear_layers(x)

        x = x.view(x.size(0), 512, 4, 4)
        x = self.conv_layers(x)
        
        kl_sum = 0
        x = x.flatten(start_dim=1)
        x = torch.concat([x, z], dim=1)
        x, kl = self.bayes_1(x)
        kl_sum += kl
        x = self.batch_norm_1(x)
        x, kl = self.bayes_2(x)
        kl_sum += kl
        x = self.batch_norm_2(x)
        x, kl = self.bayes_3(x)
        kl_sum += kl
        x = self.batch_norm_3(x)

        x = self.activation_function(x)

        return x.reshape(x.size(0), 32, 32), kl_sum
    


class vessel_DGBaNR(torch.nn.Module): # R for reparametrization
    def __init__(self, input_size, img_size, activation_function, pre_trained_base=False):
        super(vessel_DGBaNR, self).__init__()

        self.base_name = 'DGBaNR_base'
        self.dict_dict_keys = {
            'linear_layers.0.weight':'linear1.mu_weight',
            'linear_layers.0.bias':'linear1.mu_bias',
            'linear_layers.2.weight':'linear2.mu_weight',
            'linear_layers.2.bias':'linear2.mu_bias',
            'linear_layers.4.weight':'linear3.mu_weight',
            'linear_layers.4.bias':'linear3.mu_bias',
            'conv_layers.0.weight':'conv1.mu_kernel',
            'conv_layers.0.bias':'conv1.mu_bias',
            'conv_layers.1.weight':'batch_norm1.weight',
            'conv_layers.1.bias':'batch_norm1.bias',
            'conv_layers.1.running_mean':'batch_norm1.running_mean',
            'conv_layers.1.running_var':'batch_norm1.running_var',
            'conv_layers.1.num_batches_tracked':'batch_norm1.num_batches_tracked',
            'conv_layers.3.weight':'conv2.mu_kernel',
            'conv_layers.3.bias':'conv2.mu_bias',
            'conv_layers.4.weight':'batch_norm2.weight',
            'conv_layers.4.bias':'batch_norm2.bias',
            'conv_layers.4.running_mean':'batch_norm2.running_mean',
            'conv_layers.4.running_var':'batch_norm2.running_var',
            'conv_layers.4.num_batches_tracked':'batch_norm2.num_batches_tracked',
            'conv_layers.6.weight':'conv3.mu_kernel',
            'conv_layers.6.bias':'conv3.mu_bias'
        }

        self.img_size = img_size

        self.linear1 = BayesLinearR(input_size, 108)
        self.linear2 = BayesLinearR(108, 972)
        self.linear3 = BayesLinearR(972, 512 * 4 * 4)

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
        
        x, kl = self.linear3(x)
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



class mini_DGBaNR(torch.nn.Module): # R for reparametrization
    def __init__(self, input_size, img_size, activation_function, pre_trained_base=False):
        super(mini_DGBaNR, self).__init__()

        self.base_name = 'mini_DGBaNR_base'

        self.img_size = img_size

        self.linear1 = LinearReparameterization(input_size, 16)

        self.conv1 = ConvTranspose2dReparameterization(
            in_channels=16,
            out_channels=16,
            kernel_size=2,
            stride=2
        )
        self.batch_norm1 = nn.BatchNorm2d(16)
        
        self.conv2 = ConvTranspose2dReparameterization(
            in_channels=16,
            out_channels=4,
            kernel_size=4,
            stride=4
        )
        self.batch_norm2 = nn.BatchNorm2d(4)
        
        self.conv3 = ConvTranspose2dReparameterization(
            in_channels=4,
            out_channels=1,
            kernel_size=4,
            stride=4
        )

        if activation_function == 'sigmoid':
            self.activation_function = torch.sigmoid
        else:
            self.activation_function = getattr(F, activation_function)

    def forward(self, x):
        kl_sum = 0

        x, kl = self.linear1(x)
        kl_sum += kl
        x = F.relu(x)
        
        x = x.view(x.size(0), 16, 1, 1)

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



class DGBaNR_2(torch.nn.Module): # R for reparametrization
    def __init__(self, input_size, img_size, activation_function, pre_trained_base=False):
        super(DGBaNR_2, self).__init__()

        self.base_name = 'DGBaNR_2_base'

        self.img_size = img_size

        self.linear_1 = LinearReparameterization(input_size, 128)
        self.linear_2 = LinearReparameterization(128, 1024)
        self.linear_3 = LinearReparameterization(1024, 8192)

        self.conv_1 = ConvTranspose2dReparameterization(
            in_channels=512,
            out_channels=256,
            kernel_size=4,
            stride=2,
            padding=0
        )
        self.batch_norm_1 = nn.BatchNorm2d(256)
        
        self.conv_2 = ConvTranspose2dReparameterization(
            in_channels=256,
            out_channels=128,
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.batch_norm_2 = nn.BatchNorm2d(128)
        
        self.conv_3 = ConvTranspose2dReparameterization(
            in_channels=128,
            out_channels=1,
            kernel_size=2,
            stride=2,
            padding=3
        )

        if activation_function == 'sigmoid':
            self.activation_function = torch.sigmoid
        else:
            self.activation_function = getattr(F, activation_function)

    def forward(self, x):
        kl_sum = 0

        x, kl = self.linear_1(x)
        kl_sum += kl
        x = F.relu(x)
        
        x, kl = self.linear_2(x)
        kl_sum += kl
        x = F.relu(x)
        
        x, kl = self.linear_3(x)
        kl_sum += kl
        x = F.relu(x)
        
        x = x.view(x.size(0), 512, 4, 4)

        x, kl = self.conv_1(x)
        kl_sum += kl
        x = self.batch_norm_1(x)
        x = F.relu(x)

        x, kl = self.conv_2(x)
        kl_sum += kl
        x = self.batch_norm_2(x)
        x = F.relu(x)
        
        x, kl = self.conv_3(x)
        kl_sum += kl

        x = self.activation_function(x).squeeze(dim=1)

        return x, kl_sum

    def random_init(self):
        for name, module in self.named_modules():
            if isinstance(module, LinearReparameterization):
                # Re-initialize mu for weights
                nn.init.kaiming_uniform_(module.mu_weight, a=0, mode='fan_in', nonlinearity='relu')
                # Re-initialize sigma for weights
                module.rho_weight.data.fill_(-5)
                
                if module.mu_bias is not None:
                    # Re-initialize mu for biases
                    module.mu_bias.data.zero_()
                    # Re-initialize sigma for biases
                    module.rho_bias.data.fill_(-5)

            elif isinstance(module, ConvTranspose2dReparameterization):
                # Re-initialize mu for weights
                nn.init.kaiming_uniform_(module.mu_kernel, a=0, mode='fan_in', nonlinearity='relu')
                # Re-initialize sigma for weights
                module.rho_kernel.data.fill_(-5)
                
                if module.mu_bias is not None:
                    # Re-initialize mu for biases
                    module.mu_bias.data.zero_()
                    # Re-initialize sigma for biases
                    module.rho_bias.data.fill_(-5)



class DGBaNR_3(torch.nn.Module): # R for reparametrization
    def __init__(self, input_size, img_size, activation_function, pre_trained_base=False):
        super(DGBaNR_2, self).__init__()

        self.base_name = 'DGBaNR_2_base'

        self.img_size = img_size

        self.linear_layers = nn.Sequential(
            nn.linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1024),
            nn.ReLU(),
            nn.Linear(1024, 8192),
            nn.ReLU(),
            nn.Linear(8192, 32768),
            nn.ReLU()
        )

        self.conv_1 = ConvTranspose2dReparameterization(
            in_channels=512,
            out_channels=256,
            kernel_size=4,
            stride=2,
            padding=0
        )
        self.batch_norm_1 = nn.BatchNorm2d(256)
        
        self.conv_2 = ConvTranspose2dReparameterization(
            in_channels=256,
            out_channels=128,
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.batch_norm_2 = nn.BatchNorm2d(128)
        
        self.conv_3 = ConvTranspose2dReparameterization(
            in_channels=128,
            out_channels=1,
            kernel_size=2,
            stride=2,
            padding=3
        )

        if activation_function == 'sigmoid':
            self.activation_function = torch.sigmoid
        else:
            self.activation_function = getattr(F, activation_function)

    def forward(self, x):
        kl_sum = 0

        x, kl = self.linear_1(x)
        kl_sum += kl
        x = F.relu(x)
        
        x, kl = self.linear_2(x)
        kl_sum += kl
        x = F.relu(x)
        
        x, kl = self.linear_3(x)
        kl_sum += kl
        x = F.relu(x)
        
        x = x.view(x.size(0), 256, 4, 4)

        x, kl = self.conv_1(x)
        kl_sum += kl
        x = self.batch_norm_1(x)
        x = F.relu(x)

        x, kl = self.conv_2(x)
        kl_sum += kl
        x = self.batch_norm_2(x)
        x = F.relu(x)
        
        x, kl = self.conv_3(x)
        kl_sum += kl

        x = self.activation_function(x).squeeze(dim=1)

        return x, kl_sum



class DGBaN5Blocks(torch.nn.Module):
    def __init__(
            self,
            img_size: int,
            arch_dict: dict
        ):
        super(DGBaN5Blocks, self).__init__()

        self.arch = arch_dict

        self.img_size = img_size

        # BLOCK 1
        self.linear_layers = arch_dict['linear_layers']

        # BLOCK 2
        self.scale_up_1 = blocks.View((4, 4))
        self.unconv_1 = arch_dict['unconv_1']

        # BLOCK 3
        self.scale_up_2 = blocks.ScaleUp(arch_dict['scale_up_2_channels'], stride=2, padding=0, output_padding=1)
        self.unconv_2 = arch_dict['unconv_2']

        # BLOCK 4
        self.scale_up_3 = blocks.ScaleUp(arch_dict['scale_up_3_channels'], stride=2, padding=0, output_padding=1)
        self.unconv_3 = arch_dict['unconv_3']

        # BLOCK 5
        self.scale_up_4 = blocks.LastConv(arch_dict['scale_up_4_channels'], kernel=5, stride=2, padding=2, output_padding=1)
        self.unconv_4 = arch_dict['unconv_4']

    def forward(self, x):
        kl_sum = 0
        x, kl = self.linear_layers(x)
        kl_sum += kl

        x, kl = self.scale_up_1(x)
        kl_sum += kl
        x, kl = self.unconv_1(x)
        kl_sum += kl

        x, kl = self.scale_up_2(x)
        kl_sum += kl
        x, kl = self.unconv_2(x)
        kl_sum += kl

        x, kl = self.scale_up_3(x)
        kl_sum += kl
        x, kl = self.unconv_3(x)
        kl_sum += kl

        x, kl = self.scale_up_4(x)
        kl_sum += kl
        x, kl = self.unconv_4(x)
        kl_sum += kl

        return x.squeeze(dim=1), kl_sum



class DGBaN4BlocksOpitm(torch.nn.Module):
    def __init__(self, input_size, img_size, activation_function, pre_trained_base=False):
        super(DGBaN4BlocksOpitm, self).__init__()
        
        self.base_name = 'DGBaN4Blocks'

        # BLOCK 1
        self.linear_1 = nn.Linear(84, 209)
        self.linear_2 = nn.Linear(209, 524)
        self.linear_3 = nn.Linear(524, 1311)
        self.linear_4 = nn.Linear(1311, 3277)
        self.linear_5 = nn.Linear(3277, 8192)

        # BLOCK 2
        self.unconv_11 = BayesConvTranspose2d(512, 512, 3, 1, 1)
        self.bn_11 = nn.BatchNorm2d(512)
        self.unconv_12 = BayesConvTranspose2d(512, 64, 1, 1, 0)
        self.bn_12 = nn.BatchNorm2d(64)
        self.unconv_13 = BayesConvTranspose2d(64, 64, 3, 1, 1)
        self.bn_13 = nn.BatchNorm2d(64)
        self.unconv_14 = BayesConvTranspose2d(64, 256, 1, 1, 0)
        self.bn_14 = nn.BatchNorm2d(256)

        self.shortcut_1 = BayesConvTranspose2d(512, 256, 1, 1, 0, bias=False)
        self.bn_shortcut_1 = nn.BatchNorm2d(256)

        self.upscale_1 = BayesConvTranspose2d(256, 256, 1, 2, 0, output_padding=1, groups=256, bias=False)
        self.bn_upscale_1 = nn.BatchNorm2d(256)

        # BLOCK 3
        self.unconv_21 = BayesConvTranspose2d(256, 256, 3, 1, 1)
        self.bn_21 = nn.BatchNorm2d(256)
        self.unconv_22 = BayesConvTranspose2d(256, 128, 3, 1, 1)
        self.bn_22 = nn.BatchNorm2d(128)
        self.unconv_24 = BayesConvTranspose2d(128, 128, 3, 1, 1)
        self.bn_24 = nn.BatchNorm2d(128)

        self.shortcut_2 = BayesConvTranspose2d(256, 128, 1, 1, 0, bias=False)
        self.bn_shortcut_2 = nn.BatchNorm2d(128)

        self.upscale_2 = BayesConvTranspose2d(128, 128, 1, 2, 0, output_padding=1, groups=128, bias=False)
        self.bn_upscale_2 = nn.BatchNorm2d(128)

        # BLOCK 4
        self.unconv_31 = BayesConvTranspose2d(128, 128, 3, 1, 1)
        self.bn_31 = nn.BatchNorm2d(128)
        self.unconv_32 = BayesConvTranspose2d(128, 64, 3, 1, 1)
        self.bn_32 = nn.BatchNorm2d(64)
        self.unconv_34 = BayesConvTranspose2d(64, 64, 3, 1, 1)
        self.bn_34 = nn.BatchNorm2d(64)

        self.shortcut_3 = BayesConvTranspose2d(128, 64, 1, 1, 0, bias=False)
        self.bn_shortcut_3 = nn.BatchNorm2d(64)

        self.unconv_4 = BayesConvTranspose2d(64, 1, 5, 2, 2, output_padding=1)

    def forward(self, x):
        kl_sum = 0

        # BLOCK 1
        # x, kl = self.linear_1(x)
        # x = F.relu(x)
        # kl_sum += kl
        # x, kl = self.linear_2(x)
        # x = F.relu(x)
        # kl_sum += kl
        # x, kl = self.linear_3(x)
        # x = F.relu(x)
        # kl_sum += kl
        # x, kl = self.linear_4(x)
        # x = F.relu(x)
        # kl_sum += kl
        # x, kl = self.linear_5(x)
        # x = F.relu(x)
        # kl_sum += kl
        
        x = F.relu(self.linear_1(x))
        x = F.relu(self.linear_2(x))
        x = F.relu(self.linear_3(x))
        x = F.relu(self.linear_4(x))
        x = F.relu(self.linear_5(x))

        z = x.view(x.size(0), 512, 4, 4)

        # BLOCK 2
        x, kl = self.unconv_11(z)
        x = F.relu(self.bn_11(x))
        kl_sum += kl
        x, kl = self.unconv_12(x)
        x = F.relu(self.bn_12(x))
        kl_sum += kl
        x, kl = self.unconv_13(x)
        x = F.relu(self.bn_13(x))
        kl_sum += kl
        x, kl = self.unconv_14(x)
        x = self.bn_14(x)
        kl_sum += kl

        z, kl = self.shortcut_1(z)
        z = self.bn_shortcut_1(z)
        kl_sum += kl

        x = F.relu(x + z)

        x, kl = self.upscale_1(x)
        z = self.bn_upscale_1(x)
        kl_sum += kl

        # BLOCK 3
        x, kl = self.unconv_21(x)
        x = F.relu(self.bn_21(x))
        kl_sum += kl
        x, kl = self.unconv_22(x)
        x = F.relu(self.bn_22(x))
        kl_sum += kl
        x, kl = self.unconv_24(x)
        x = self.bn_24(x)
        kl_sum += kl

        z, kl = self.shortcut_2(z)
        z = self.bn_shortcut_2(z)
        kl_sum += kl

        x = F.relu(x + z)

        x, kl = self.upscale_2(x)
        z = self.bn_upscale_2(x)
        kl_sum += kl

        # BLOCK 4
        x, kl = self.unconv_31(x)
        x = F.relu(self.bn_31(x))
        kl_sum += kl
        x, kl = self.unconv_32(x)
        x = F.relu(self.bn_32(x))
        kl_sum += kl
        x, kl = self.unconv_34(x)
        x = self.bn_34(x)
        kl_sum += kl

        z, kl = self.shortcut_3(z)
        z = self.bn_shortcut_3(z)
        kl_sum += kl

        x = F.relu(x + z)

        x, kl = self.unconv_4(x)
        kl_sum += kl

        return x.squeeze(dim=1), kl_sum



class DGBaN4BlocksLittleBayes(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(DGBaN4BlocksLittleBayes, self).__init__()
        
        self.base_name = 'DGBaN4Blocks'

        # BLOCK 1
        self.linear_1 = nn.Linear(84, 209)
        self.linear_2 = nn.Linear(209, 524)
        self.linear_3 = nn.Linear(524, 1311)
        self.linear_4 = nn.Linear(1311, 3277)
        self.linear_5 = nn.Linear(3277, 8192)

        # BLOCK 2
        self.unconv_11 = nn.ConvTranspose2d(512, 512, 3, 1, 1)
        self.bn_11 = nn.BatchNorm2d(512)
        self.unconv_12 = nn.ConvTranspose2d(512, 64, 1, 1, 0)
        self.bn_12 = nn.BatchNorm2d(64)
        self.unconv_13 = nn.ConvTranspose2d(64, 64, 3, 1, 1)
        self.bn_13 = nn.BatchNorm2d(64)
        self.unconv_14 = nn.ConvTranspose2d(64, 256, 1, 1, 0)
        self.bn_14 = nn.BatchNorm2d(256)

        self.shortcut_1 = nn.ConvTranspose2d(512, 256, 1, 1, 0, bias=False)
        self.bn_shortcut_1 = nn.BatchNorm2d(256)

        self.upscale_1 = BayesConvTranspose2d(256, 256, 1, 2, 0, output_padding=1, groups=256, bias=False)
        self.bn_upscale_1 = nn.BatchNorm2d(256)

        # BLOCK 3
        self.unconv_21 = nn.ConvTranspose2d(256, 256, 3, 1, 1)
        self.bn_21 = nn.BatchNorm2d(256)
        self.unconv_22 = nn.ConvTranspose2d(256, 128, 3, 1, 1)
        self.bn_22 = nn.BatchNorm2d(128)
        self.unconv_24 = nn.ConvTranspose2d(128, 128, 3, 1, 1)
        self.bn_24 = nn.BatchNorm2d(128)

        self.shortcut_2 = nn.ConvTranspose2d(256, 128, 1, 1, 0, bias=False)
        self.bn_shortcut_2 = nn.BatchNorm2d(128)

        self.upscale_2 = BayesConvTranspose2d(128, 128, 1, 2, 0, output_padding=1, groups=128, bias=False)
        self.bn_upscale_2 = nn.BatchNorm2d(128)

        # BLOCK 4
        self.unconv_31 = nn.ConvTranspose2d(128, 128, 3, 1, 1)
        self.bn_31 = nn.BatchNorm2d(128)
        self.unconv_32 = nn.ConvTranspose2d(128, 64, 3, 1, 1)
        self.bn_32 = nn.BatchNorm2d(64)
        self.unconv_34 = nn.ConvTranspose2d(64, 64, 3, 1, 1)
        self.bn_34 = nn.BatchNorm2d(64)

        self.shortcut_3 = nn.ConvTranspose2d(128, 64, 1, 1, 0, bias=False)
        self.bn_shortcut_3 = nn.BatchNorm2d(64)

        self.unconv_4 = BayesConvTranspose2d(64, 1, 5, 2, 2, output_padding=1)

    def forward(self, x):
        kl_sum = 0

        # BLOCK 1
        # x, kl = self.linear_1(x)
        # x = F.relu(x)
        # kl_sum += kl
        # x, kl = self.linear_2(x)
        # x = F.relu(x)
        # kl_sum += kl
        # x, kl = self.linear_3(x)
        # x = F.relu(x)
        # kl_sum += kl
        # x, kl = self.linear_4(x)
        # x = F.relu(x)
        # kl_sum += kl
        # x, kl = self.linear_5(x)
        # x = F.relu(x)
        # kl_sum += kl
        
        x = F.relu(self.linear_1(x))
        x = F.relu(self.linear_2(x))
        x = F.relu(self.linear_3(x))
        x = F.relu(self.linear_4(x))
        x = F.relu(self.linear_5(x))

        z = x.view(x.size(0), 512, 4, 4)

        # BLOCK 2
        x = self.unconv_11(z)
        x = F.relu(self.bn_11(x))

        x = self.unconv_12(x)
        x = F.relu(self.bn_12(x))
        
        x = self.unconv_13(x)
        x = F.relu(self.bn_13(x))
        
        x = self.unconv_14(x)
        x = self.bn_14(x)
        
        z = self.shortcut_1(z)
        z = self.bn_shortcut_1(z)

        x = F.relu(x + z)

        x, kl = self.upscale_1(x)
        z = self.bn_upscale_1(x)
        kl_sum += kl

        # BLOCK 3
        x = self.unconv_21(x)
        x = F.relu(self.bn_21(x))
        
        x = self.unconv_22(x)
        x = F.relu(self.bn_22(x))
        
        x = self.unconv_24(x)
        x = self.bn_24(x)
        
        z = self.shortcut_2(z)
        z = self.bn_shortcut_2(z)

        x = F.relu(x + z)

        x, kl = self.upscale_2(x)
        z = self.bn_upscale_2(x)
        kl_sum += kl

        # BLOCK 4
        x = self.unconv_31(x)
        x = F.relu(self.bn_31(x))
        
        x = self.unconv_32(x)
        x = F.relu(self.bn_32(x))
        
        x = self.unconv_34(x)
        x = self.bn_34(x)
        
        z = self.shortcut_3(z)
        z = self.bn_shortcut_3(z)

        x = F.relu(x + z)

        x, kl = self.unconv_4(x)
        kl_sum += kl

        return x.squeeze(dim=1), kl_sum



class DGBaNConv17(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(DGBaNConv17, self).__init__()

        self.linear_layers = blocks.NLinearNormAct(28, 1024, 5)

        self.conv_layers = blocks.BayesSequential(
            blocks.BottleNeck(256, 256),
            blocks.Conv2x3x3NormAct(256, 128, shortcut=blocks.Conv1x1BnReLU(256, 128)),
            blocks.ResidualAdd(
                blocks.BayesSequential(
                    blocks.ScaleUp(128, 128, 2, 0, 1),
                    blocks.Conv3x3BnReLU(128, 128)
                ),
                blocks.ConvNormAct(128, 128, 1, 2, 0, 1)
            ),
            blocks.Conv2x3x3NormAct(128, 64, shortcut=blocks.Conv1x1BnReLU(128, 64)),
            blocks.ResidualAdd(
                blocks.BayesSequential(
                    blocks.ScaleUp(64, 64, 2, 0, 1),
                    blocks.Conv3x3BnReLU(64, 64)
                ),
                blocks.ConvNormAct(64, 64, 1, 2, 0, 1)
            ),
            blocks.Conv2x3x3NormAct(64, 32, shortcut=blocks.Conv1x1BnReLU(64, 32)),
            blocks.ResidualAdd(
                blocks.BayesSequential(
                    blocks.ScaleUp(32, 32, 2, 0, 1),
                    blocks.Conv3x3BnReLU(32, 32)
                ),
                blocks.ConvNormAct(32, 32, 1, 2, 0, 1)
            ),
            blocks.Conv2x3x3NormAct(32, 16, shortcut=blocks.Conv1x1BnReLU(32, 16)),
            blocks.LastConv(16, 5, 2, 2, 1)
        )

    def forward(self, x):
        kl_sum = 0

        x, kl = self.linear_layers(x)
        kl_sum += kl

        x = x.view(x.size(0), 256, 2, 2)

        x, kl = self.conv_layers(x)
        kl_sum += kl
        
        return x.squeeze(dim=1).reshape(x.size(0), 1024), kl_sum
    


class DGBaNLinear22(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(DGBaNLinear22, self).__init__()

        self.linear_layers = blocks.BayesSequential(
            blocks.NLinearNormAct(28, 1024, 5),
            blocks.ResidualAdd(
                blocks.BayesSequential(
                    blocks.LinearNormAct(1024, 256, bias=False),
                    blocks.LinearNormAct(256, 256),
                    blocks.LinearNormAct(256, 1024, bias=False),
                )
            ),
            blocks.ResidualAdd(
                blocks.BayesSequential(
                    blocks.LinearNormAct(1024, 1024),
                    blocks.LinearNormAct(1024, 1024)
                )
            ),
            blocks.ResidualAdd(
                blocks.BayesSequential(
                    blocks.LinearNormAct(1024, 1024),
                    blocks.LinearNormAct(1024, 1024)
                )
            ),
            blocks.ResidualAdd(
                blocks.BayesSequential(
                    blocks.LinearNormAct(1024, 1024),
                    blocks.LinearNormAct(1024, 1024)
                )
            ),
            blocks.ResidualAdd(
                blocks.BayesSequential(
                    blocks.LinearNormAct(1024, 1024),
                    blocks.LinearNormAct(1024, 1024)
                )
            ),
            blocks.ResidualAdd(
                blocks.BayesSequential(
                    blocks.LinearNormAct(1024, 1024),
                    blocks.LinearNormAct(1024, 1024)
                )
            ),
            blocks.ResidualAdd(
                blocks.BayesSequential(
                    blocks.LinearNormAct(1024, 1024),
                    blocks.LinearNormAct(1024, 1024)
                )
            ),
            blocks.ResidualAdd(
                blocks.BayesSequential(
                    blocks.LinearNormAct(1024, 1024),
                    blocks.LinearNormAct(1024, 1024)
                )
            )
        )

    def forward(self, x):
        x, kl = self.linear_layers(x)
        x = x.view(x.size(0), 32, 32)
        
        return x, kl



class OnePixel(nn.Module):
    def __init__(self, *args, **kwargs):
        super(OnePixel, self).__init__()
        
        # self.conv_layers = blocks.BayesSequential(
        #     blocks.ConvNormAct(1, 2, 5, 2, 1, transpose=False),
        #     blocks.ConvNormAct(2, 4, 3, 2, 1, transpose=False),
        #     blocks.ConvNormAct(4, 8, 3, 2, 1, transpose=False),
        #     blocks.ConvNormAct(8, 16, 3, 2, 1, transpose=False)
        # )
        
        self.linear_1 = BayesLinear(36, 18)
        self.linear_2 = BayesLinear(18, 9)
        self.linear_3 = BayesLinear(9, 1)
        
    def forward(self, x):
        # z = self.conv_layers(true_target)

        kl_sum = 0
        x, kl = self.linear_1(x)
        kl_sum += kl
        x = F.relu(x)
        x, kl = self.linear_2(x)
        kl_sum += kl
        x = F.relu(x)
        x, kl = self.linear_3(x)
        kl_sum += kl
        
        return x, kl_sum



class DGBaN1024(nn.Module):
    def __init__(self, *args, **kwargs):
        super(DGBaN1024, self).__init__()
        self.pixels = nn.ModuleList([OnePixel() for _ in range(1024)])

    def forward(self, X, true_target):
        x = torch.zeros((X.size(0), 1024), device=X.device, dtype=X.dtype)
        kl = torch.zeros((X.size(0), 1024), device=X.device, dtype=X.dtype)

        for i in range(1024):
            xi, kli = self.pixels[i](X, true_target[:, i])
            x[:, i] = xi.squeeze()
            kl[:, i] = kli.squeeze()

        return x, kl



class DGBaNConv25(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(DGBaNConv25, self).__init__()

        self.linear_layers = blocks.NLinearNormAct(15, 1024, 5)

        self.conv_layers = blocks.BayesSequential(
            blocks.BottleNeck(256, 256),
            blocks.Conv2x3x3NormAct(256, 256),
            blocks.Conv2x3x3NormAct(256, 128, shortcut=blocks.Conv1x1BnReLU(256, 128)),
            blocks.ResidualAdd(
                blocks.BayesSequential(
                    blocks.ScaleUp(128, 128, 2, 0, 1),
                    blocks.Conv3x3BnReLU(128, 128)
                ),
                blocks.ConvNormAct(128, 128, 1, 2, 0, 1)
            ),
            blocks.Conv2x3x3NormAct(128, 128),
            blocks.Conv2x3x3NormAct(128, 64, shortcut=blocks.Conv1x1BnReLU(128, 64)),
            blocks.ResidualAdd(
                blocks.BayesSequential(
                    blocks.ScaleUp(64, 64, 2, 0, 1),
                    blocks.Conv3x3BnReLU(64, 64)
                ),
                blocks.ConvNormAct(64, 64, 1, 2, 0, 1)
            ),
            blocks.Conv2x3x3NormAct(64, 64),
            blocks.Conv2x3x3NormAct(64, 32, shortcut=blocks.Conv1x1BnReLU(64, 32)),
            blocks.ResidualAdd(
                blocks.BayesSequential(
                    blocks.ScaleUp(32, 32, 2, 0, 1),
                    blocks.Conv3x3BnReLU(32, 32)
                ),
                blocks.ConvNormAct(32, 32, 1, 2, 0, 1)
            ),
            blocks.Conv2x3x3NormAct(32, 32),
            blocks.Conv2x3x3NormAct(32, 16, shortcut=blocks.Conv1x1BnReLU(32, 16)),
            blocks.LastConv(16, 5, 2, 2, 1)
        )

    def forward(self, x):
        kl_sum = 0

        x, kl = self.linear_layers(x)
        kl_sum += kl

        x = x.view(x.size(0), 256, 2, 2)

        x, kl = self.conv_layers(x)
        kl_sum += kl
        
        return x.squeeze(dim=1), kl_sum
    

class DGBaNConv33(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(DGBaNConv33, self).__init__()

        self.linear_layers = blocks.NLinearNormAct(28, 2048, 6)

        self.conv_layers = blocks.BayesSequential(
            blocks.BottleNeck(512, 512),
            blocks.Conv2x3x3NormAct(512, 512),
            blocks.Conv2x3x3NormAct(512, 512),
            blocks.Conv2x3x3NormAct(512, 256, shortcut=blocks.Conv1x1BnReLU(512, 256)),
            blocks.ResidualAdd(
                blocks.BayesSequential(
                    blocks.ScaleUp(256, 256, 2, 0, 1),
                    blocks.Conv3x3BnReLU(256, 256)
                ),
                blocks.ConvNormAct(256, 256, 1, 2, 0, 1)
            ),
            blocks.Conv2x3x3NormAct(256, 256),
            blocks.Conv2x3x3NormAct(256, 256),
            blocks.Conv2x3x3NormAct(256, 128, shortcut=blocks.Conv1x1BnReLU(256, 128)),
            blocks.ResidualAdd(
                blocks.BayesSequential(
                    blocks.ScaleUp(128, 128, 2, 0, 1),
                    blocks.Conv3x3BnReLU(128, 128)
                ),
                blocks.ConvNormAct(128, 128, 1, 2, 0, 1)
            ),
            blocks.Conv2x3x3NormAct(128, 128),
            blocks.Conv2x3x3NormAct(128, 128),
            blocks.Conv2x3x3NormAct(128, 64, shortcut=blocks.Conv1x1BnReLU(128, 64)),
            blocks.ResidualAdd(
                blocks.BayesSequential(
                    blocks.ScaleUp(64, 64, 2, 0, 1),
                    blocks.Conv3x3BnReLU(64, 64)
                ),
                blocks.ConvNormAct(64, 64, 1, 2, 0, 1)
            ),
            blocks.Conv2x3x3NormAct(64, 64),
            blocks.Conv2x3x3NormAct(64, 64),
            blocks.Conv2x3x3NormAct(64, 32, shortcut=blocks.Conv1x1BnReLU(64, 32)),
            blocks.LastConv(32, 5, 2, 2, 1)
        )

    def forward(self, x):
        kl_sum = 0

        x, kl = self.linear_layers(x)
        kl_sum += kl

        x = x.view(x.size(0), 512, 2, 2)

        x, kl = self.conv_layers(x)
        kl_sum += kl
        
        return x.squeeze(dim=1), kl_sum
    

