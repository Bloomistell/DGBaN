from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F



class LinearNormAct(nn.Sequential):
    def __init__(
        self,
        input: int,
        output: int,
        n_layers: int,
        norm: nn.Module = None,
        act: nn.Module = nn.ReLU,
        **kwargs
    ):
        factor = (output/input)**(1/n_layers)

        seq = []
        for i in range(n_layers):
            seq.append(nn.Linear(input * factor**i), int(input * factor**(i+1)))
            seq.append(norm(int(input * factor**(i+1))))
            seq.append(act())

        super().__init__(*seq)


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        input: int,
        output: int,
        kernel_size: int,
        stride: int,
        padding: int,
        norm: nn.Module = nn.BatchNorm2d,
        act: nn.Module = nn.ReLU,
        **kwargs
    ):

        super().__init__(
            nn.Conv2d(
                input,
                output,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            norm(output),
            act(),
        )

Conv1x1BnReLU = partial(ConvNormAct, kernel_size=1, stride=1, padding=0)
Conv3x3BnReLU = partial(ConvNormAct, kernel_size=3, stride=1, padding=1)


class ResidualAdd(nn.Module):
    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.block(x)
        if self.shortcut:
            res = self.shortcut(res)
        x += res
        return x


class BottleNeck(nn.Sequential):
    def __init__(self, input: int, output: int, reduction: int = 4):
        reduced_input = output // reduction
        super().__init__(
            nn.Sequential(
                ResidualAdd(
                    nn.Sequential(
                        # wide -> narrow
                        Conv1x1BnReLU(input, reduced_input),
                        # narrow -> narrow
                        Conv3x3BnReLU(reduced_input, reduced_input),
                        # narrow -> wide
                        Conv1x1BnReLU(reduced_input, output, act=nn.Identity),
                    ),
                    shortcut=Conv1x1BnReLU(input, output)
                    if input != output
                    else None,
                ),
                nn.ReLU(),
            )
        )


class LinearBottleNeck(nn.Sequential):
    def __init__(self, input: int, output: int, reduction: int = 4):
        reduced_input = output // reduction
        super().__init__(
            nn.Sequential(
                ResidualAdd(
                    nn.Sequential(
                        # wide -> narrow
                        Conv1x1BnReLU(input, reduced_input),
                        # narrow -> narrow
                        Conv3x3BnReLU(reduced_input, reduced_input),
                        # narrow -> wide
                        Conv1x1BnReLU(reduced_input, output, act=nn.Identity),
                    ),
                    shortcut=Conv1x1BnReLU(input, output)
                    if input != output
                    else None,
                ),
            )
        )
        

class LastConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_size):
        super(LastConv, self).__init__()
        self.output_size = output_size
        self.convT = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
    
    def forward(self, x):
        return self.convT(x, self.output_size)