import sys
sys.path.append('..')

import argparse

import pickle

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn

import matplotlib.pyplot as plt
import numpy as np

from DGBaN import ring_dataset, randomized_ring_dataset, energy_randomized_ring_dataset, DGBaNR, big_DGBaNR

from IPython.display import clear_output



def train_model(
    data_type='energy_random',
    model_name='DGBaNR',
    activation_function='sigmoid',
    model_path='',
    save_model='../save_model/',
    data_size=64_000,
    epochs=200,
    batch_size=64,
    optim_name='Adam',
    loss_name='nll_loss',
    num_mc=10,
    lr=1e-2,
    train_fraction=0.8,
    save_interval=20,
    random_seed=42
):
    writer = SummaryWriter(f'../runs/{model_name}_{activation_function}_{data_type}_{data_size}_{batch_size}_{optim_name}_{loss_name}_{num_mc}/')


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.device(device)

    ### load dataset ###
    N = 32
    if data_type == 'simple':
        data_gen = simple_ring_dataset(N=N)
    elif data_type == 'random':
        data_gen = randomized_ring_dataset(N=N)
    elif data_type == 'energy_random':
        data_gen = energy_randomized_ring_dataset(N=N)

    train_loader, test_loader = data_gen.generate_dataset(data_size=data_size, batch_size=batch_size, seed=random_seed, device=device)


    ### load model ###
    if model_name == 'DGBaNR':
        generator = DGBaNR(data_gen.n_features, img_size=N).to(device)
    elif model_name == 'big_DGBaNR':
        generator = big_DGBaNR(data_gen.n_features, N, activation_function).to(device)

    example_data, _ = next(iter(test_loader))

    writer.add_graph(generator, example_data)

    if model_path: # use a pretrained model
        generator.load_state_dict(torch.load(model_path))


    ### set optimizer and loss ###
    optimizer = getattr(optim, optim_name)(generator.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    loss_fn = getattr(F, loss_name)


    ### training loop ###
    generator.train()

    train_steps = len(train_loader)
    test_steps = len(test_loader)

    for epoch in range(epochs):
        sum_loss = 0.
        for i, (X, target) in tqdm(enumerate(train_loader), f'EPOCH {epoch+1}', train_steps, leave=False, unit='batch'):
            optimizer.zero_grad()

            ### it seems to better capture the distribution probability wih only one mc (pretty logic actually) ###
            # pred_ = []
            # kl_ = []
            # for _ in range(num_mc): # extract several samples from the model
            #     pred, kl = generator(X)
            #     pred_.append(pred)
            #     kl_.append(kl)

            # pred = torch.mean(torch.stack(pred_), dim=0)
            # kl = torch.mean(torch.stack(kl_), dim=0)

            pred, kl = generator(X)

            loss = loss_fn(pred, target) + (kl / batch_size)
            sum_loss += loss.item()

            loss.backward()
            optimizer.step()

            if i % (train_steps // 10) == 0 and i != 0:
                writer.add_scalar('training loss', sum_loss / (train_steps // 10), epoch * train_steps + i)
                sum_loss = 0.
                
        scheduler.step()

        sum_loss = 0.
        with torch.no_grad(): # evaluate model on test data
            for i, (X, target) in enumerate(test_loader):
                # pred_ = []
                # kl_ = []
                # for _ in range(num_mc):
                #     pred, kl = generator(X)
                #     pred_.append(pred)
                #     kl_.append(kl)

                # pred = torch.mean(torch.stack(pred_), dim=0)
                # kl = torch.mean(torch.stack(kl_), dim=0)

                pred, kl = generator(X)

                loss = loss_fn(pred, target) + (kl / batch_size)
                sum_loss += loss.item()

                if i % (test_steps // 10) == 0 and i != 0:
                    writer.add_scalar('testing loss', sum_loss / (test_steps // 10), epoch * test_steps + i)
                    sum_loss = 0.
        
        if save_model and epoch % save_interval == 0 and epoch != 0:
            torch.save(generator.state_dict(), save_model + f'{model_name}_{activation_function}_{data_type}_{data_size}_{batch_size}_{optim_name}_{loss_name}_{num_mc}.pt')



if __name__ == "__main__" :

    parser = argparse.ArgumentParser(description='Train different generative architectures on simplistic rings.', argument_default=argparse.SUPPRESS)
    parser.add_argument('-t', '--data_type', type=str, help="Type of data")
    parser.add_argument('-n', '--model_name', type=str, help="name of the model")
    parser.add_argument('-m', '--model_path', type=str, help="Pretained model path")
    parser.add_argument('-s', '--save_model', type=str, help="Pretained model path")
    parser.add_argument('-d', '--data_size', type=int, help="Number of events to train on")
    parser.add_argument('-e', '--epochs', type=int, help="Number of epochs to train for")
    parser.add_argument('-b', '--batch_size', type=int, help="Batch size")
    parser.add_argument('-o', '--optim_name', type=str, help="name of the optimizer")
    parser.add_argument('-l', '--loss_name', type=str, help="name of the loss function")
    parser.add_argument('-mc', '--num_mc', type=int, help="Number of Monte Carlo runs during training")
    parser.add_argument('-lr', '--lr', type=float, help="Learning rate")
    parser.add_argument('-f', '--train_fraction', type=float, help="Fraction of data used for training")
    parser.add_argument('-i', '--save_interval', type=int, help="Save network state every <save_interval> iterations")
    parser.add_argument('-r', '--random_seed', type=int, help="Random seed")

    args = parser.parse_args()

    for arg_name, arg in vars(args).items():
        print(f'   {arg_name}: {arg}')

    train_model(**vars(args))

    # python3 training/train_bayesian_model.py -t energy_random -n DGBaNR -e 200 -d 64000 -b 64 -o Adam -l mse_loss -mc 20
