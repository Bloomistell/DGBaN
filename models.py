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

