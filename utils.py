import os

import numpy as np

import torch
import torch.nn as nn
import torch.functional as F
import torch.distributions as dist



def count_params(model):
    params = model.state_dict()
    
    S = 0
    for weights_and_biases in params.values():
        S += np.prod(weights_and_biases.size())

    return int(S)



def save_txt(txt, path):
    with open(path, 'w') as f:
        f.write(txt)



class GaussianMixture():
    def __init__(self, mean, std, weight):
        self.mean = mean
        self.std = std
        self.weight = weight

        self.gaussians = [[dist.Normal(m, s) for m, s in zip(mean_row, std_row)] for mean_row, std_row in zip(mean, std)]
        
    def sample(self, num_samples):
        num_components = len(self.gaussians[0])
        num_rows = len(self.gaussians)
        
        # initialize tensor to hold the samples
        samples = torch.empty((num_rows, num_samples), dtype=torch.float)
        
        # for each row
        for i in range(num_rows):
            # sample component indices according to the mixture weights
            indices = dist.Categorical(self.weight[i]).sample((num_samples,))
            
            # for each component, draw the samples from the appropriate gaussian
            for j in range(num_components):
                mask = (indices == j)
                samples[i, mask] = self.gaussians[i][j].sample((mask.sum(),))
        
        return samples