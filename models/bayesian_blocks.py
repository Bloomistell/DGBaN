from functools import partial
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from bayesian_torch.layers.variational_layers.linear_variational import LinearReparameterization as BayesLinear
from bayesian_torch.layers.variational_layers.conv_variational import ConvTranspose2dReparameterization as BayesConvTranspose2d



class BayesIdentity(nn.Module):
    def __init__(self):
        super(BayesIdentity, self).__init__()

    def forward(self, x):
        return x, 0


class BayesSequential(nn.Module):
    def __init__(self, *args):
        super(BayesSequential, self).__init__()
        self.bayes_modules = nn.ModuleList(args)

    def forward(self, x):
        kl_sum = 0
        for module in self.bayes_modules:
            x, kl = module(x)
            kl_sum += kl

        return x, kl


class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape
        self.scale = shape[0] * shape[1]
    
    def forward(self, x):
        return x.view(x.size(0), x.size(1) // self.scale, *self.shape), 0 # 0 for compatibility with other scale ups


class ScaleUp(nn.Module):
    def __init__(self, channels, stride, padding, output_padding):
        super(ScaleUp, self).__init__()
        # NOTE: you have to modify the source code for ConvTranspose2dReparameterization because groups and dilation are inverted
        self.conv_t = BayesConvTranspose2d(channels, channels, 1, stride, padding, output_padding=output_padding, groups=channels)

    def forward(self, x):
        return self.conv_t(x)


class LinearNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_layers: int,
        norm: nn.Module = None,
        act: nn.Module = nn.ReLU,
        **kwargs
    ):
        super(LinearNormAct, self).__init__()

        self.n_layers = n_layers

        factor = (out_channels/in_channels)**(1/n_layers)

        self.linears = []
        self.norms_acts = []
        for i in range(n_layers):
            self.linears.append(BayesLinear(int(in_channels * factor**i), int(in_channels * factor**(i+1))))
            if norm:
                self.norms_acts.append(nn.Sequential(norm(int(in_channels * factor**(i+1))), act()))
            else:
                self.norms_acts.append(act())
        
        self.linears = nn.ModuleList(self.linears)
        self.norms_acts = nn.ModuleList(self.norms_acts)
        self.act = act()

    def forward(self, x):
        kl_sum = 0
        for i in range(self.n_layers):
            x, kl = self.linears[i](x)
            kl_sum += kl
            x = self.norms_acts[i](x)

        return x, kl_sum


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,

        kernel_size: int,
        stride: int,
        padding: int,
        norm: nn.Module = nn.BatchNorm2d,
        act: nn.Module = nn.ReLU,
        **kwargs
    ):
        super(ConvNormAct, self).__init__()

        self.conv = BayesConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
        self.norm_act = nn.Sequential(norm(out_channels), act())

    def forward(self, x):
        x, kl = self.conv(x)
        x = self.norm_act(x)
        return x, kl


Conv1x1BnReLU = partial(ConvNormAct, kernel_size=1, stride=1, padding=0)
Conv3x3BnReLU = partial(ConvNormAct, kernel_size=3, stride=1, padding=1)


class ResidualAdd(nn.Module):
    def __init__(self, block: nn.Module, shortcut: nn.Module = None):
        super().__init__()
        self.block = block
        self.shortcut = shortcut
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kl_sum = 0
        res = x
        x, kl = self.block(x)
        kl_sum += kl

        if self.shortcut:
            res, kl = self.shortcut(res)
        x += res
        kl_sum += kl
        return x, kl_sum


class BottleNeck(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, reduction: int = 4):
        reduced_channels = out_channels // reduction
        super(BottleNeck, self).__init__()
        self.block = ResidualAdd(
            BayesSequential(
                ConvNormAct(in_channels, out_channels, 3, 1, 1),
                # wide -> narrow
                Conv1x1BnReLU(out_channels, reduced_channels),
                # narrow -> narrow
                Conv3x3BnReLU(reduced_channels, reduced_channels),
                # narrow -> wide
                Conv1x1BnReLU(reduced_channels, out_channels, act=nn.Identity),
            ),
            shortcut=Conv1x1BnReLU(in_channels, out_channels)
            if in_channels != out_channels
            else None,
        )
        
    def forward(self, x):
        x, kl = self.block(x)
        x = F.relu(x)
        return x, kl


class LinearBottleNeck(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, reduction: int = 4):
        reduced_in_channels = out_channels // reduction
        super(LinearBottleNeck, self).__init__()
        self.block = ResidualAdd(
            BayesSequential(
                ConvNormAct(in_channels, out_channels, 3, 1, 1),
                # wide -> narrow
                Conv1x1BnReLU(out_channels, reduced_in_channels),
                # narrow -> narrow
                Conv3x3BnReLU(reduced_in_channels, reduced_in_channels),
                # narrow -> wide
                Conv1x1BnReLU(reduced_in_channels, out_channels, act=nn.Identity),
            ),
            shortcut=Conv1x1BnReLU(in_channels, out_channels)
            if in_channels != out_channels
            else None,
        )
        
    def forward(self, x):
        x, kl = self.block(x)
        x = F.relu(x)
        return x, kl


class Conv3x11x3NormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super(Conv3x11x3NormAct, self).__init__()
        self.block = ResidualAdd(
            BayesSequential(
                ConvNormAct(in_channels, out_channels, 3, 1, 1),
                ConvNormAct(out_channels, out_channels, (3, 1), 1, (1, 0)),
                ConvNormAct(out_channels, out_channels, (1, 3), 1, (0, 1)),
                ConvNormAct(out_channels, out_channels, 3, 1, 1, act=nn.Identity)
            ),
            shortcut=Conv1x1BnReLU(in_channels, out_channels)
            if in_channels != out_channels
            else None,
        )

    def forward(self, x):
        x, kl = self.block(x)
        x = F.relu(x)
        return x, kl

        
class Conv3x3x3NormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super(Conv3x3x3NormAct, self).__init__()
        self.block = ResidualAdd(
            BayesSequential(
                ConvNormAct(in_channels, out_channels, 3, 1, 1),
                ConvNormAct(out_channels, out_channels, 3, 1, 1),
                ConvNormAct(out_channels, out_channels, 3, 1, 1, act=nn.Identity)
            ),
            shortcut=Conv1x1BnReLU(in_channels, out_channels)
            if in_channels != out_channels
            else None,
        )

    def forward(self, x):
        x, kl = self.block(x)
        x = F.relu(x)
        return x, kl


class LastConv(nn.Module):
    def __init__(self, in_channels, kernel, stride, padding, output_padding):
        super(LastConv, self).__init__()
        self.conv_t = BayesConvTranspose2d(in_channels, 1, kernel, stride, padding, output_padding=output_padding)
    
    def forward(self, x):
        x, kl = self.conv_t(x)
        return x, kl