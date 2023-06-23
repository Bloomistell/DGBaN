from functools import partial
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from bayesian_torch.layers.variational_layers.linear_variational import LinearReparameterization as BayesLinear
from bayesian_torch.layers.variational_layers.conv_variational import (
    ConvTranspose2dReparameterization as BayesConvTranspose2d,
    Conv2dReparameterization as BayesConv2d
)



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


class BayesSequential(nn.Module):
    def __init__(self, *args):
        super(BayesSequential, self).__init__()
        self.bayes_modules = nn.ModuleList(args)

    def forward(self, x, mul: torch.Tensor = None):
        kl_sum = 0
        for module in self.bayes_modules:
            x, kl = module(x, mul)
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
    def __init__(self, in_channels, out_channels, stride, padding, output_padding):
        super(ScaleUp, self).__init__()
        # NOTE: you have to modify the source code for ConvTranspose2dReparameterization because groups and dilation are inverted
        self.conv_t = BayesConvTranspose2d(in_channels, out_channels, 1, stride, padding, output_padding=output_padding, groups=out_channels)

    def forward(self, x, mul: torch.Tensor = None):
        return self.conv_t(x)


class LinearNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        act: nn.Module = nn.ReLU,
        **kwargs
    ):
        super(LinearNormAct, self).__init__()

        self.linear = BayesLinear(in_channels, out_channels, bias=bias)
        self.act = act()
        
    def forward(self, x):
        x, kl = self.linear(x)
        x = self.act(x)
        return x, kl


class NLinearNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_layers: int,
        act: nn.Module = nn.ReLU,
        last_layer: bool = False,
        **kwargs
    ):
        super(NLinearNormAct, self).__init__()

        self.n_layers = n_layers

        factor = (out_channels/in_channels)**(1/n_layers)

        self.linear_act = []
        for i in range(n_layers-1):
            self.linear_act.append(BayesLinear(int(in_channels * factor**i), int(in_channels * factor**(i+1))))
            self.linear_act.append(act())
            penultimate = int(in_channels * factor**(i+1))
            
        self.linear_act.append(BayesLinear(penultimate, out_channels))
        if not last_layer:
            self.linear_act.append(act())
        
        self.linear_act = nn.ModuleList(self.linear_act)

    def forward(self, x):
        kl_sum = 0
        for i in range(0, self.n_layers * 2, 2):
            x, kl = self.linear_act[i](x)
            kl_sum += kl
            x = self.linear_act[i+1](x)

        return x, kl_sum


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,

        kernel_size: int,
        stride: int,
        padding: int,
        output_padding: int = 0,
        norm: nn.Module = nn.BatchNorm2d,
        act: nn.Module = nn.ReLU,
        transpose: bool = True,
        mul: bool = False,
        **kwargs
    ):
        super(ConvNormAct, self).__init__()

        self.out_channels = out_channels

        if transpose:
            self.conv = BayesConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding
                )
        else:
            self.conv = BayesConv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding
                )
        self.norm = norm(out_channels)
        self.act = act()

    def forward(self, x, mul: torch.Tensor = None):
        if mul is None:
            x, kl = self.conv(x)
            x = self.act(self.norm(x))

        else:
            mul = torch.cat((mul,) * (self.out_channels // 28), dim=1)

            x, kl = self.conv(x)
            x = self.act(self.norm(x) * mul)
        
        return x, kl


Conv1x1BnReLU = partial(ConvNormAct, kernel_size=1, stride=1, padding=0)
Conv3x3BnReLU = partial(ConvNormAct, kernel_size=3, stride=1, padding=1)


class ResidualAdd(nn.Module):
    def __init__(self, block: nn.Module, shortcut: nn.Module = None):
        super().__init__()
        self.block = block
        self.shortcut = shortcut

    def forward(self, x: torch.Tensor, mul: torch.Tensor = None) -> torch.Tensor:
        kl_sum = 0
        res = x
        x, kl = self.block(x, mul)
        kl_sum += kl

        if self.shortcut:
            res, kl = self.shortcut(res)
        x = x + res
        kl_sum += kl
        return x, kl_sum


class ResidualMul(nn.Module):
    def __init__(self, block: nn.Module, shortcut: nn.Module = None):
        super().__init__()
        self.block = block
        self.shortcut = shortcut

    def forward(self, x: torch.Tensor, mul: torch.Tensor = None) -> torch.Tensor:
        kl_sum = 0
        res = x
        x, kl = self.block(x, mul)
        kl_sum += kl

        if self.shortcut:
            res, kl = self.shortcut(res)
        x = x * res
        kl_sum += kl
        return x, kl_sum
    

class ResidualAddMul(nn.Module):
    def __init__(self, block: nn.Module, shortcut: nn.Module = None):
        super().__init__()
        self.block = block
        self.shortcut = shortcut

    def forward(self, x: torch.Tensor, mul: torch.Tensor = None) -> torch.Tensor:
        kl_sum = 0
        res = x
        x, kl = self.block(x, mul)
        kl_sum += kl

        if self.shortcut:
            res, kl = self.shortcut(res)
        x = x * res + res
        kl_sum += kl
        return x, kl_sum


class BottleNeck(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, shortcut: nn.Module = None, reduction: int = 4, residual: nn.Module = ResidualAdd):
        reduced_channels = out_channels // reduction
        super(BottleNeck, self).__init__()

        self.out_channels = out_channels

        self.block = residual(
            BayesSequential(
                Conv1x1BnReLU(out_channels, reduced_channels),
                Conv3x3BnReLU(reduced_channels, reduced_channels),
                Conv1x1BnReLU(reduced_channels, out_channels, act=nn.Identity),
            ),
            shortcut,
        )
        
    def forward(self, x, mul: torch.Tensor = None):
        x, kl = self.block(x, mul)
        x = F.relu(x)

        return x, kl


class LinearBottleNeck(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, shortcut: nn.Module = None, reduction: int = 4, residual: nn.Module = ResidualAdd):
        reduced_in_channels = out_channels // reduction
        super(LinearBottleNeck, self).__init__()
        
        self.out_channels = out_channels

        self.block = residual(
            BayesSequential(
                Conv1x1BnReLU(out_channels, reduced_in_channels),
                Conv3x3BnReLU(reduced_in_channels, reduced_in_channels),
                Conv1x1BnReLU(reduced_in_channels, out_channels, act=nn.Identity),
            ),
            shortcut,
        )
        
    def forward(self, x, mul: torch.Tensor = None):
        x, kl = self.block(x, mul)
        x = F.relu(x)

        return x, kl


class Conv3x11x3NormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, shortcut: nn.Module = None, residual: nn.Module = ResidualAdd):
        super(Conv3x11x3NormAct, self).__init__()
        
        self.out_channels = out_channels

        self.block = residual(
            BayesSequential(
                ConvNormAct(in_channels, in_channels, (3, 1), 1, (1, 0)),
                ConvNormAct(in_channels, out_channels, (1, 3), 1, (0, 1), act=nn.Identity)
            ),
            shortcut,
        )

    def forward(self, x, mul: torch.Tensor = None):
        x, kl = self.block(x, mul)
        x = F.relu(x)

        return x, kl

        
class Conv2x3x3NormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, shortcut: nn.Module = None, residual: nn.Module = ResidualAdd):
        super(Conv2x3x3NormAct, self).__init__()

        self.out_channels = out_channels

        self.block = residual(
            BayesSequential(
                ConvNormAct(in_channels, in_channels, 3, 1, 1),
                ConvNormAct(in_channels, out_channels, 3, 1, 1, act=nn.Identity)
            ),
            shortcut,
        )
    
    def forward(self, x, mul: torch.Tensor = None):
        x, kl = self.block(x, mul)
        x = F.relu(x)

        return x, kl


class LastConv(nn.Module):
    def __init__(self, in_channels, kernel, stride, padding, output_padding):
        super(LastConv, self).__init__()
        self.conv_t = BayesConvTranspose2d(in_channels, 1, kernel, stride, padding, output_padding=output_padding)

    def forward(self, x, mul=None): # mul for consistancy among the blocks
        x, kl = self.conv_t(x)
        return x, kl