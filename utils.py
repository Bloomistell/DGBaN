import os

import numpy as np

import torch
import torch.nn as nn
import torch.functional as F



def count_params(model):
    params = model.state_dict()
    
    S = 0
    for weights_and_biases in params.values():
        S += np.prod(weights_and_biases.size())

    return int(S)


def save_txt(txt, path):
    with open(path, 'w') as f:
        f.write(txt)