import torch
import torch.nn as nn
import torch.nn.functional as F

from bayesian_torch.layers.variational_layers.linear_variational import LinearReparameterization
from bayesian_torch.layers.variational_layers.conv_variational import ConvTranspose2dReparameterization



prior_mu = 0.0
prior_sigma = 1.0
posterior_mu_init = 0.0
posterior_rho_init = -3.0


def BayesLinearR(n_input, n_output):
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

        self.linear_1 = LinearReparameterization(input_size, 16)
        
        self.linear_2 = LinearReparameterization(16, 64)

        self.linear_3 = LinearReparameterization(64, 256)

        self.conv_1 = ConvTranspose2dReparameterization(
            in_channels=64,
            out_channels=64,
            kernel_size=2,
            stride=2
        )
        self.batch_norm_1 = nn.BatchNorm2d(64)
        
        self.conv_2 = ConvTranspose2dReparameterization(
            in_channels=64,
            out_channels=16,
            kernel_size=2,
            stride=2
        )
        self.batch_norm_2 = nn.BatchNorm2d(16)
        
        self.conv_3 = ConvTranspose2dReparameterization(
            in_channels=16,
            out_channels=4,
            kernel_size=2,
            stride=4
        )
        self.batch_norm_3 = nn.BatchNorm2d(4)
        
        self.conv_4 = ConvTranspose2dReparameterization(
            in_channels=4,
            out_channels=1,
            kernel_size=8,
            stride=2,
            padding=1
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
        
        x = x.view(x.size(0), 64, 1, 1)

        x, kl = self.conv_1(x)
        kl_sum += kl
        # x = self.batch_norm_1(x)
        x = F.relu(x)

        x, kl = self.conv_2(x)
        kl_sum += kl
        # x = self.batch_norm_2(x)
        x = F.relu(x)
        
        x, kl = self.conv_3(x)
        kl_sum += kl
        # x = self.batch_norm_3(x)
        x = F.relu(x)
        
        x, kl = self.conv_4(x)
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