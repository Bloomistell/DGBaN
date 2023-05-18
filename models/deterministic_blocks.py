from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F



class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape
        self.scale = shape[0] * shape[1]
    
    def forward(self, x):
        return x.view(x.size(0), x.size(1) // self.scale, *self.shape)


class ScaleUp(nn.Module):
    def __init__(self, channels, stride, padding, output_size):
        super(ScaleUp, self).__init__()
        self.conv_t = nn.ConvTranspose2d(channels, channels, 1, stride, padding, groups=channels, bias=False)
        self.output_size = output_size

    def forward(self, x):
        return self.conv_t(x, self.output_size)


class LinearNormAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_layers: int,
        norm: nn.Module = None,
        act: nn.Module = nn.ReLU,
        **kwargs
    ):
        factor = (out_channels/in_channels)**(1/n_layers)

        seq = []
        for i in range(n_layers-1):
            seq.append(nn.Linear(int(in_channels * factor**i), int(in_channels * factor**(i+1))))
            if norm:
                seq.append(norm(int(in_channels * factor**(i+1))))
            seq.append(act())
            
        seq.append(nn.Linear(int(in_channels * factor**(i+1)), out_channels))
        if norm:
            seq.append(norm(out_channels))
        seq.append(act())

        super().__init__(*seq)


class ConvNormAct(nn.Sequential):
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
        last_layer: bool = False,
        **kwargs
    ):
        if not last_layer:
            super().__init__(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding
                ),
                norm(out_channels),
                act(),
            )
        else:
            super().__init__(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding
                )
            )


Conv1x1BnReLU = partial(ConvNormAct, kernel_size=1, stride=1, padding=0)
Conv3x3BnReLU = partial(ConvNormAct, kernel_size=3, stride=1, padding=1)


class ResidualAdd(nn.Module):
    def __init__(self, block: nn.Module, shortcut: nn.Module = None):
        super().__init__()
        self.block = block
        self.shortcut = shortcut
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.block(x)
        if self.shortcut:
            res = self.shortcut(res)
        x = x + res
        return x


class BottleNeck(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, shortcut: nn.Module = None, last_layer: bool = False, reduction: int = 4):
        self.name = 'bottle_neck'
        reduced_channels = in_channels // reduction
        super().__init__(
            nn.Sequential(
                ResidualAdd(
                    nn.Sequential(
                        # wide -> narrow
                        Conv1x1BnReLU(in_channels, reduced_channels),
                        # narrow -> narrow
                        Conv3x3BnReLU(reduced_channels, reduced_channels),
                        # narrow -> wide
                        Conv1x1BnReLU(reduced_channels, out_channels, act=nn.Identity, last_layer=last_layer),
                    ),
                    shortcut if shortcut else None,
                ),
                nn.ReLU() if not last_layer else nn.Identity(),
            )
        )


class LinearBottleNeck(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, shortcut: nn.Module = None, last_layer: bool = False, reduction: int = 4):
        self.name = 'linear_bottle_neck'
        reduced_in_channels = in_channels // reduction
        super().__init__(
            nn.Sequential(
                ResidualAdd(
                    nn.Sequential(
                        # wide -> narrow
                        Conv1x1BnReLU(in_channels, reduced_in_channels),
                        # narrow -> narrow
                        Conv3x3BnReLU(reduced_in_channels, reduced_in_channels),
                        # narrow -> wide
                        Conv1x1BnReLU(reduced_in_channels, out_channels, act=nn.Identity, last_layer=last_layer),
                    ),
                    shortcut if shortcut else None,
                ),
            )
        )
        

class Conv3x11x3NormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, shortcut: nn.Module = None, last_layer: bool = False):
        self.name = 'conv_3x11x3'
        super().__init__(
            nn.Sequential(
                ResidualAdd(
                    nn.Sequential(
                        ConvNormAct(in_channels, in_channels, (3, 1), 1, (1, 0)),
                        ConvNormAct(in_channels, out_channels, (1, 3), 1, (0, 1), act=nn.Identity, last_layer=last_layer)
                    ),
                    shortcut if shortcut else None,
                ),
                nn.ReLU() if not last_layer else nn.Identity(),
            )
        )

        
class Conv2x3x3NormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, shortcut: nn.Module = None, last_layer: bool = False):
        self.name = 'conv_3x3x3'
        super().__init__(
            nn.Sequential(
                ResidualAdd(
                    nn.Sequential(
                        ConvNormAct(in_channels, in_channels, 3, 1, 1),
                        ConvNormAct(in_channels, out_channels, 3, 1, 1, act=nn.Identity, last_layer=last_layer)
                    ),
                    shortcut if shortcut else None,
                ),
                nn.ReLU() if not last_layer else nn.Identity(),
            )
        )


class LastConv(nn.Module):
    def __init__(self, in_channels, kernel, stride, padding, output_size):
        super(LastConv, self).__init__()
        self.conv_t = nn.ConvTranspose2d(in_channels, 1, kernel, stride, padding)
        self.output_size = output_size
    
    def forward(self, x):
        return self.conv_t(x, self.output_size)