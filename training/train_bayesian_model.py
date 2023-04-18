import sys
sys.path.append('..')

import argparse

import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn

import matplotlib.pyplot as plt
import numpy as np

from DGBaN.datasets.generate_simple_dataset import ring_dataset, randomized_ring_dataset
from DGBaN.models.bayesian_models import DGBaNR

from IPython.display import clear_output



def train_model(
    data_type='Simple',
    model_name='Conv',
    model_path='',
    save_model='',
    data_size=10_000,
    epochs=10,
    batch_size=64,
    num_mc=10,
    lr=1e-2,
    num_workers=8,
    train_fraction=0.8,
    save_interval=20,
    output_dir="",
    random_seed=42
):
    writer = SummaryWriter('../runs')


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.device(device)


    ### load dataset ###
    N = 32
    if data_type == 'Simple':
        dataset_generator = simple_ring_dataset(N=N)
    elif data_type == 'Random':
        dataset_generator = randomized_ring_dataset(N=N)

    train_loader, test_loader = dataset_generator.generate_dataset(data_size=data_size, batch_size=batch_size, seed=random_seed, device=device)


    ### load model ###
    if model_name == 'DGBaNR':
        generator = DGBaNR(dataset_generator.n_features, img_size=N).to(device)

    example_data, _ = next(iter(test_loader))

    writer.add_graph(generator, example_data)

    if model_path: # use a pretrained model
        generator.load_state_dict(torch.load(model_path))


    ### set optimizer ###
    optimizer = optim.RMSprop(generator.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)


    ### training loop ###
    train_loss = []
    test_loss = []
    generator.train()

    train_steps = len(train_loader)
    test_steps = len(test_loader)

    for epoch in range(epochs):
        sum_loss = 0.
        for i, (X, target) in enumerate(train_loader):
            optimizer.zero_grad()

            pred_ = []
            kl_ = []
            for _ in range(num_mc): # extract several samples from the model
                pred, kl = generator(X)
                pred_.append(pred)
                kl_.append(kl)

            pred = torch.mean(torch.stack(pred_), dim=0)
            kl = torch.mean(torch.stack(kl_), dim=0)
            loss = F.mse_loss(pred, target) + (kl / batch_size)
            train_loss.append(loss)
            sum_loss += loss.item()

            loss.backward()
            optimizer.step()

            if i % 100 == 0 and i != 0:
                writer.add_scalar('training loss', sum_loss / 100, epoch * train_steps + i)
                sum_loss = 0.
                
        scheduler.step()

        with torch.no_grad(): # evaluate model on test data
            total_loss = 0
            for i, (X, target) in enumerate(test_loader):
                pred_ = []
                kl_ = []
                for _ in range(num_mc):
                    pred, kl = generator(X)
                    pred_.append(pred)
                    kl_.append(kl)

                pred = torch.mean(torch.stack(pred_), dim=0)
                kl = torch.mean(torch.stack(kl_), dim=0)
                loss = F.mse_loss(pred, target) + (kl / batch_size)
                test_loss.append(loss)
                sum_loss += loss.item()

                if i % 100 == 0 and i != 0:
                    writer.add_scalar('testing loss', sum_loss / 100, epoch * test_steps + i)
                    sum_loss = 0.
        
        print(f'EPOCH {epoch+1}: train loss - {train_loss[-1]:.2g}, test loss - {test_loss[-1]:.2g}')
        
        if save_model and epoch % save_interval == 0 and epoch != 0:
            torch.save(generator.state_dict(), save_model)

    generator.eval()
    return generator, dataset_generator


if __name__ == "__main__" :

    parser = argparse.ArgumentParser(description='Train different generative architectures on simplistic rings.')
    parser.add_argument('-t', '--data_type', type=str, help="Type of data", default ="", required=False)
    parser.add_argument('-n', '--model_name', type=str, help="name of the model", default ="", required=False)
    parser.add_argument('-m', '--model_path', type=str, help="Pretained model path", default ="", required=False)
    parser.add_argument('-s', '--save_model', type=str, help="Pretained model path", default ="", required=False)
    parser.add_argument('-d', '--data_size', type=int, help="Number of events to train on", default=10_000, required=False)
    parser.add_argument('-e', '--epochs', type=int, help="Number of epochs to train for", default=10, required=False)
    parser.add_argument('-b', '--batch_size', type=int, help="Batch size", default=64, required=False)
    parser.add_argument('-mc', '--num_mc', type=int, help="Number of Monte Carlo runs during training", default=10, required=False)
    parser.add_argument('-lr', '--lr', type=float, help="Learning rate", default=1e-2, required=False)
    parser.add_argument('-j', '--num_workers', type=int, help="Number of CPUs for loading data", default=8, required=False)
    parser.add_argument('-f', '--train_fraction', type=float, help="Fraction of data used for training", default=0.8, required=False)
    parser.add_argument('-i', '--save_interval', type=int, help="Save network state every <save_interval> iterations", default=20, required=False)
    parser.add_argument('-r', '--random_seed', type=int, help="Random seed", default=42, required=False)

    args = parser.parse_args()
    print(vars(args))
    
    train_model(**vars(args))

    # python3 training/train_bayesian_model.py -t Random -n DGBaNR -s ../save_model/first_DGBaN.pt -e 200 -d 6400 -b 64
