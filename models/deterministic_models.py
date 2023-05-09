import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import (
    Linear,
    Sequential,
    ReLU,
    BatchNorm1d,
    Sigmoid,
    ConvTranspose2d,
    BatchNorm2d
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
        print(x.size(0))

        img = self.conv_net(x.view(x.size(0), 512, 4, 4)).squeeze()
        return img



class DGBaNR_base(torch.nn.Module):
    def __init__(self, input_size, img_size, activation_function):
        super(DGBaNR_base, self).__init__()

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

        if activation_function == 'sigmoid':
            self.activation_function = torch.sigmoid
        else:
            self.activation_function = getattr(F, activation_function)

    def forward(self, x):
        x = self.linear_layers(x)

        x = x.view(x.size(0), 512, 4, 4)
        x = self.conv_layers(x)
        
        x = self.activation_function(x)

        return x.squeeze(dim=1)
    


class DGBaNR_2_base(torch.nn.Module):
    def __init__(self, input_size, img_size, activation_function, pre_trained_base=False):
        super(DGBaNR_2_base, self).__init__()

        self.base_name = 'DGBaNR_2_base'
        self.img_size = img_size

        self.linear_1 = nn.Linear(input_size, 16)
        self.linear_2 = nn.Linear(16, 64)

        self.conv_1 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=64,
            kernel_size=2,
            stride=2
        )
        self.batch_norm_1 = nn.BatchNorm2d(64)
        
        self.conv_2 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=16,
            kernel_size=2,
            stride=2
        )
        self.batch_norm_2 = nn.BatchNorm2d(16)
        
        self.conv_3 = nn.ConvTranspose2d(
            in_channels=16,
            out_channels=4,
            kernel_size=2,
            stride=4
        )
        self.batch_norm_3 = nn.BatchNorm2d(4)
        
        self.conv_4 = nn.ConvTranspose2d(
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
        x = F.relu(self.linear_1(x))
        x = F.relu(self.linear_2(x))
        
        x = x.view(x.size(0), 64, 1, 1)

        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
        x = F.relu(self.conv_3(x))
        x = self.conv_4(x)

        x = self.activation_function(x).squeeze(dim=1)

        return x
