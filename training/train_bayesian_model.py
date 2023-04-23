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

from DGBaN import ring_dataset, randomized_ring_dataset, energy_randomized_ring_dataset, pattern_randomized_ring_dataset, DGBaNR, big_DGBaNR

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
    mean_training=True,
    std_training=True,
    train_fraction=0.8,
    save_interval=1,
    random_seed=42
):
    print(f"""TRAINING SUMMARY:
        data_type (-t): {data_type}
        model_name (-n): {model_name}
        activation_function (-a): {activation_function}
        model_path (-m): {model_path}
        save_model (-s): {save_model}
        data_size (-d): {data_size}
        epochs (-e): {epochs}
        batch_size (-b): {batch_size}
        optim_name (-o): {optim_name}
        loss_name (-l): {loss_name}
        num_mc (-mc): {num_mc}
        lr (-lr): {lr}
        mean_training (-mt): {mean_training}
        std_training (-st): {std_training}
        train_fraction (-f): {train_fraction}
        save_interval (-i): {save_interval}
        random_seed (-r): {random_seed}
        """)

    # proceed = None
    # yes = ['', 'yes', 'y']
    # no = ['no', 'n']
    # while proceed not in yes and proceed not in no:
    #     proceed = input('Proceed ([y]/n)? ').lower()
    #     if proceed in no:
    #         print('Aborting...')
    #         return

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
    elif data_type == 'pattern_random':
        data_gen = pattern_randomized_ring_dataset(N=N)

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

    
    ### set accuracy variables
    n_samples = 1000
    feature = torch.Tensor(data_gen.scaler.transform(np.array([[15, 15, 9.4, 3.6]])))
    features = [feature.to(device) for i in range(n_samples)]

    true_prob = torch.Tensor(data_gen.gaussian_from_features(15, 15, 9.4, 3.6)).to(device)


    ### training loop ###
    generator.train()

    train_steps = len(train_loader)
    test_steps = len(test_loader)

    for epoch in range(epochs):
        sum_loss = 0.
        if mean_training: # here the goal is to train the mean of the neurone's gaussians, so we need num_mc > 1
            for i, (X, target) in tqdm(enumerate(train_loader), f'EPOCH {epoch+1}', train_steps, leave=False, unit='batch'):
                if mean_training:
                    optimizer.zero_grad()

                    preds = []
                    kls = []
                    for i in range(num_mc): # extract several samples from the model
                        pred, kl = generator(X)
                        preds.append(pred)
                        kls.append(kl)

                    pred = torch.mean(torch.stack(preds), dim=0)
                    kl = torch.mean(torch.stack(kls), dim=0)

                    loss = loss_fn(pred, target) + (kl / batch_size)
                    sum_loss += loss.item()

                    loss.backward()
                    optimizer.step()

                if std_training:
                    optimizer.zero_grad()

                    pred, kl = generator(X)

                    loss = loss_fn(pred, target) + (kl / batch_size)
                    sum_loss += loss.item()

                    loss.backward()
                    optimizer.step()

                if i % 100 == 0 and i != 0:
                    writer.add_scalar('training loss', sum_loss / 100, epoch * train_steps + i)
                    sum_loss = 0.
                
        scheduler.step()

        sum_loss = 0.
        with torch.no_grad(): # evaluate model on test data
            for i, (X, target) in tqdm(enumerate(test_loader), 'Validation', test_steps, leave=False, unit='batch'):
                pred, kl = generator(X)

                loss = loss_fn(pred, target) + (kl / batch_size)
                sum_loss += loss.item()

            writer.add_scalar('testing loss', sum_loss / test_steps, epoch * test_steps + i)
            sum_loss = 0.

            # getting the predictions for the base features
            pred_rings = torch.zeros((n_samples, 32, 32)).to(device)
            for i, tensor in enumerate(features):
                pred_rings[i] += generator(tensor)[0].squeeze()

            pred_ring = pred_rings.sum(axis=0)
            pred_prob = pred_ring / pred_ring.max()

            writer.add_scalar('accuracy', 1 - (true_prob - pred_prob).mean(), epoch)
        

        if save_model and epoch % save_interval == 0 and epoch != 0:
            torch.save(generator.state_dict(), save_model + f'{model_name}_{activation_function}_{data_type}_{data_size}_{batch_size}_{optim_name}_{loss_name}_{num_mc}.pt')



if __name__ == "__main__" :

    parser = argparse.ArgumentParser(description='Train different generative architectures on simplistic rings.', argument_default=argparse.SUPPRESS)
    parser.add_argument('-t', '--data_type', type=str, help="Type of data")
    parser.add_argument('-n', '--model_name', type=str, help="name of the model")
    parser.add_argument('-a', '--activation_function', type=str, help="activation function")
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

    train_model(**vars(args))

    # python3 training/train_bayesian_model.py -t energy_random -n DGBaNR -e 200 -d 64000 -b 64 -o Adam -l mse_loss -mc 20
