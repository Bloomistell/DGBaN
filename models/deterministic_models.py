from collections import OrderedDict
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

import models.deterministic_blocks as blocks



class LinearGenerator(torch.nn.Module):
    def __init__(self, input_size, img_size):
        super(LinearGenerator, self).__init__()

        self.img_size = img_size

        self.network = nn.Sequential(
            nn.Linear(input_size, 12),
            nn.ReLU(),

            nn.Linear(12, 36),
            nn.BatchNorm1d(36),
            nn.ReLU(),

            nn.Linear(36, 108),
            nn.BatchNorm1d(108),
            nn.ReLU(),

            nn.Linear(108, 324),
            nn.BatchNorm1d(324),
            nn.ReLU(),

            nn.Linear(324, img_size * img_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.network(x)
        return x.view(x.size(0), self.img_size, self.img_size)



class ConvGenerator(torch.nn.Module):
    def __init__(self, input_size, img_size):
        super(ConvGenerator, self).__init__()

        self.img_size = img_size

        self.neural_net = nn.Sequential(
            nn.Linear(input_size, 512 * 4 * 4),
            nn.ReLU(),
        )

        self.conv_net = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
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

        self.img_size = img_size

        self.linear_1 = nn.Linear(6, 128)
        self.linear_2 = nn.Linear(128, 1024)
        self.linear_3 = nn.Linear(1024, 8192)

        self.unconv_1 = nn.ConvTranspose2d(
            in_channels=512,
            out_channels=256,
            kernel_size=4,
            stride=2,
            padding=0
        )
        self.batch_norm_1 = nn.BatchNorm2d(256)
        
        self.unconv_2 = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.batch_norm_2 = nn.BatchNorm2d(128)

        self.unconv_3 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=2,
            stride=2,
            padding=1
        )
        self.batch_norm_3 = nn.BatchNorm2d(64)

        self.unconv_4 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=2
        )

        if activation_function == 'sigmoid':
            self.activation_function = torch.sigmoid
        else:
            self.activation_function = getattr(F, activation_function)

    def forward(self, x):
        x = F.relu(self.linear_1(x))
        x = F.relu(self.linear_2(x))
        x = F.relu(self.linear_3(x))
        
        x = x.view(x.size(0), 512, 4, 4)

        x = F.relu(self.batch_norm_1(self.unconv_1(x)))
        x = F.relu(self.batch_norm_2(self.unconv_2(x)))
        x = F.relu(self.batch_norm_3(self.unconv_3(x)))
        x = self.unconv_4(x)

        x = self.activation_function(x).squeeze(dim=1)

        return x



class DGBaNR_3_base(torch.nn.Module):
    def __init__(self, input_size, img_size, activation_function, pre_trained_base=False):
        super(DGBaNR_3_base, self).__init__()

        self.img_size = img_size

        self.linear_1 = nn.Linear(6, 128)
        self.linear_2 = nn.Linear(128, 256)
        self.linear_3 = nn.Linear(256, 512)
        self.linear_4 = nn.Linear(512, 1024)
        self.linear_5 = nn.Linear(1024, 2048)


        self.unconv_1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.bn_1 = nn.BatchNorm2d(256)

        self.unconv_2 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.bn_2 = nn.BatchNorm2d(256)
        self.unconv_3 = nn.ConvTranspose2d(256, 32, kernel_size=1, stride=1, padding=0)
        self.bn_3 = nn.BatchNorm2d(32)
        self.unconv_4 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn_4 = nn.BatchNorm2d(32)
        self.unconv_5 = nn.ConvTranspose2d(32, 256, kernel_size=1, stride=1, padding=0)
        self.bn_5 = nn.BatchNorm2d(256)

        self.unconv_6 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=0)
        self.bn_6 = nn.BatchNorm2d(128)
        self.unconv_7 = nn.ConvTranspose2d(128, 128, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.bn_7 = nn.BatchNorm2d(128)
        self.unconv_8 = nn.ConvTranspose2d(128, 128, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.bn_8 = nn.BatchNorm2d(128)

        self.unconv_9 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1)
        self.bn_9 = nn.BatchNorm2d(64)
        self.unconv_10 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn_10 = nn.BatchNorm2d(64)
        self.unconv_11 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn_11 = nn.BatchNorm2d(64)

        self.unconv_12 = nn.ConvTranspose2d(64, 1, kernel_size=5, stride=2, padding=3)


        if activation_function == 'sigmoid':
            self.activation_function = torch.sigmoid
        else:
            self.activation_function = getattr(F, activation_function)

    def forward(self, x):
        x = F.relu(self.linear_1(x))
        x = F.relu(self.linear_2(x))
        x = F.relu(self.linear_3(x))
        x = F.relu(self.linear_4(x))
        x = F.relu(self.linear_5(x))
        
        x = x.view(x.size(0), 512, 2, 2)

        x = F.relu(self.bn_1(self.unconv_1(x)))

        z = F.relu(self.bn_2(self.unconv_2(x)))
        x = F.relu(self.bn_3(self.unconv_3(z)))
        x = F.relu(self.bn_4(self.unconv_4(x)))
        x = F.relu(self.bn_5(self.unconv_5(x)) + z)

        z = F.relu(self.bn_6(self.unconv_6(x)))
        x = F.relu(self.bn_7(self.unconv_7(z)))
        x = F.relu(self.bn_8(self.unconv_8(x)) + z)

        z = F.relu(self.bn_9(self.unconv_9(x)))
        x = F.relu(self.bn_10(self.unconv_10(z)))
        x = F.relu(self.bn_11(self.unconv_11(x)) + z)

        x = self.activation_function(self.unconv_12(x, output_size=(32, 32))).squeeze(dim=1)

        return x



class DGBase4Blocks(torch.nn.Module):
    def __init__(
            self,
            img_size: int,
            arch_dict: dict
        ):
        super(DGBase4Blocks, self).__init__()

        self.img_size = img_size
        self.arch = arch_dict


        # BLOCK 1
        self.linear_layers = arch_dict['linear_layers']

        # BLOCK 2
        self.scale_up_1 = blocks.View((8, 8))
        self.unconv_1 = arch_dict['unconv_1']

        # BLOCK 3
        self.scale_up_2 = blocks.ScaleUp(arch_dict['scale_up_2_channels'], stride=2, padding=0, output_size=(16, 16))
        self.unconv_2 = arch_dict['unconv_2']

        # BLOCK 4
        self.scale_up_3 = blocks.ScaleUp(arch_dict['scale_up_3_channels'], stride=2, padding=0, output_size=(32, 32))
        self.unconv_3 = arch_dict['unconv_3']

    def forward(self, x):
        x = self.linear_layers(x)

        x = self.scale_up_1(x)
        x = self.unconv_1(x)

        x = self.scale_up_2(x)
        x = self.unconv_2(x)

        x = self.scale_up_3(x)
        x = self.unconv_3(x).squeeze(dim=1)

        if self.arch['sigmoid']:
            x = torch.sigmoid(x)

        return x



class DGBase5Blocks(torch.nn.Module):
    def __init__(
            self,
            img_size: int,
            arch_dict: dict
        ):
        super(DGBase5Blocks, self).__init__()

        self.arch = arch_dict

        self.img_size = img_size

        # BLOCK 1
        self.linear_layers = arch_dict['linear_layers']

        # BLOCK 2
        self.scale_up_1 = blocks.View((2, 2))
        self.unconv_1 = arch_dict['unconv_1']

        # BLOCK 3
        self.scale_up_2 = blocks.ScaleUp(arch_dict['scale_up_2_channels'], stride=4, padding=0, output_size=(5, 5))
        self.unconv_2 = arch_dict['unconv_2']

        # BLOCK 4
        self.scale_up_3 = blocks.ScaleUp(arch_dict['scale_up_3_channels'], stride=3, padding=0, output_size=(14, 14))
        self.unconv_3 = arch_dict['unconv_3']

        # BLOCK 5
        self.scale_up_4 = blocks.LastConv(arch_dict['scale_up_4_channels'], kernel=5, stride=2, padding=0, output_size=(32, 32))
        self.unconv_4 = arch_dict['unconv_4']

    def forward(self, x):
        x = self.linear_layers(x)

        x = self.scale_up_1(x)
        x = self.unconv_1(x)

        x = self.scale_up_2(x)
        x = self.unconv_2(x)

        x = self.scale_up_3(x)
        x = self.unconv_3(x)

        x = self.scale_up_4(x)
        x = self.unconv_4(x)

        return x.squeeze(dim=1)



class DGBase6Blocks(torch.nn.Module):
    def __init__(
            self,
            img_size: int,
            arch_dict: dict
        ):
        super(DGBase6Blocks, self).__init__()

        self.img_size = img_size

        # BLOCK 1
        self.linear_layers = arch_dict['linear_layers']

        # BLOCK 2
        self.scale_up_1 = blocks.View((2, 2))
        self.unconv_1 = arch_dict['unconv_1']

        # BLOCK 3
        self.scale_up_2 = blocks.ScaleUp(arch_dict['scale_up_2_channels'], stride=2, padding=0, output_size=(4, 4))
        self.unconv_2 = arch_dict['unconv_2']

        # BLOCK 4
        self.scale_up_3 = blocks.ScaleUp(arch_dict['scale_up_3_channels'], stride=2, padding=0, output_size=(8, 8))
        self.unconv_3 = arch_dict['unconv_3']

        # BLOCK 5
        self.scale_up_4 = blocks.ScaleUp(arch_dict['scale_up_4_channels'], stride=2, padding=0, output_size=(16, 16))
        self.unconv_4 = arch_dict['unconv_4']
        
        # BLOCK 6
        self.scale_up_5 = blocks.LastConv(arch_dict['scale_up_5_channels'], kernel=5, stride=2, padding=2, output_size=(32, 32))
        self.unconv_5 = arch_dict['unconv_5']

    def forward(self, x):
        x = self.linear_layers(x)

        x = self.unconv_1(self.scale_up_1(x))
        x = self.unconv_2(self.scale_up_2(x))
        x = self.unconv_3(self.scale_up_3(x))
        x = self.unconv_4(self.scale_up_4(x))
        x = self.unconv_5(self.scale_up_5(x))

        return x.squeeze(dim=1)



class DGBase7Blocks(torch.nn.Module):
    def __init__(
            self,
            img_size: int,
            arch_dict: dict
        ):
        super(DGBase7Blocks, self).__init__()

        self.img_size = img_size

        # BLOCK 1
        self.linear_layers = arch_dict['linear_layers']

        # BLOCK 2
        self.scale_up_1 = blocks.View((2, 2))
        self.unconv_1 = arch_dict['unconv_1']

        # BLOCK 3
        self.scale_up_2 = blocks.ScaleUp(arch_dict['scale_up_2_channels'], stride=2, padding=0, output_size=(3, 3))
        self.unconv_2 = arch_dict['unconv_2']

        # BLOCK 4
        self.scale_up_3 = blocks.ScaleUp(arch_dict['scale_up_3_channels'], stride=2, padding=0, output_size=(5, 5))
        self.unconv_3 = arch_dict['unconv_3']

        # BLOCK 5
        self.scale_up_4 = blocks.ScaleUp(arch_dict['scale_up_4_channels'], stride=2, padding=1, output_size=(7, 7))
        self.unconv_4 = arch_dict['unconv_4']
        
        # BLOCK 6
        self.scale_up_5 = blocks.ScaleUp(arch_dict['scale_up_5_channels'], stride=2, padding=2, output_size=(10, 10))
        self.unconv_5 = arch_dict['unconv_5']
        
        # BLOCK 7
        self.scale_up_6 = blocks.ScaleUp(arch_dict['scale_up_6_channels'], stride=2, padding=2, output_size=(16, 16))
        self.unconv_6 = arch_dict['unconv_6']
        

        self.last_conv = blocks.LastConv(arch_dict['last_conv_channels'], kernel=5, stride=2, padding=2, output_size=(32, 32))

    def forward(self, x):
        x = self.linear_layers(x)

        x = self.unconv_1(self.scale_up_1(x))
        x = self.unconv_2(self.scale_up_2(x))
        x = self.unconv_3(self.scale_up_3(x))
        x = self.unconv_4(self.scale_up_4(x))
        x = self.unconv_5(self.scale_up_5(x))
        x = self.unconv_6(self.scale_up_6(x))
        x = self.last_conv(x)

        return x.squeeze(dim=1)



class DGBaseConv17(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(DGBaseConv17, self).__init__()

        self.linear_layers = blocks.LinearNormAct(28, 1024, 5)

        self.conv_layers = nn.Sequential(
            blocks.BottleNeck(256, 256),
            blocks.Conv2x3x3NormAct(256, 128, shortcut=blocks.Conv1x1BnReLU(256, 128)),
            blocks.ResidualAdd(
                nn.Sequential(
                    blocks.ScaleUp(128, 2, 0, (4, 4)),
                    blocks.Conv3x3BnReLU(128, 128)
                ),
                blocks.ConvNormAct(128, 128, 1, 2, 0, 1)
            ),
            blocks.Conv2x3x3NormAct(128, 64, shortcut=blocks.Conv1x1BnReLU(128, 64)),
            blocks.ResidualAdd(
                nn.Sequential(
                    blocks.ScaleUp(64, 2, 0, (8, 8)),
                    blocks.Conv3x3BnReLU(64, 64)
                ),
                blocks.ConvNormAct(64, 64, 1, 2, 0, 1)
            ),
            blocks.Conv2x3x3NormAct(64, 32, shortcut=blocks.Conv1x1BnReLU(64, 32)),
            blocks.ResidualAdd(
                nn.Sequential(
                    blocks.ScaleUp(32, 2, 0, (16, 16)),
                    blocks.Conv3x3BnReLU(32, 32)
                ),
                blocks.ConvNormAct(32, 32, 1, 2, 0, 1)
            ),
            blocks.Conv2x3x3NormAct(32, 16, shortcut=blocks.Conv1x1BnReLU(32, 16)),
            blocks.LastConv(16, 5, 2, 2, (32, 32))
        )

    def forward(self, x):
        x = self.linear_layers(x)

        x = x.view(x.size(0), 256, 2, 2)

        x = self.conv_layers(x)
        
        return x.squeeze(dim=1)
    
