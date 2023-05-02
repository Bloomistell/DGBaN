import sys
sys.path.append('..')
import os
import time
import argparse
import pickle
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import numpy as np

from DGBaN import (
    generate_simple_dataset,
    deterministic_models
)



def train_model(
    data_type='energy_random',
    model_name='DGBaNR',
    activation_function='sigmoid',
    pre_trained=True,
    pre_trained_id='max',
    save_path='../save_data',
    data_size=64000,
    epochs=200,
    batch_size=64,
    optim_name='Adam',
    loss_name='mse_loss',
    lr=1e-2,
    train_fraction=0.8,
    save_interval=1,
    random_seed=42
):
    print(
    f"""
    TRAINING SUMMARY:
        data_type (-t): {data_type}
        model_name (-n): {model_name}
        activation_function (-a): {activation_function}
        pre_trained (-pt): {pre_trained}
        pre_trained_id (-id): {pre_trained_id}
        save_path (-s): {save_path}
        data_size (-d): {data_size}
        epochs (-e): {epochs}
        batch_size (-b): {batch_size}
        optim_name (-o): {optim_name}
        loss_name (-l): {loss_name}
        lr (-lr): {lr}
        train_fraction (-f): {train_fraction}
        save_interval (-i): {save_interval}
        random_seed (-r): {random_seed}
        """
    )

    # proceed = None
    # yes = ['', 'yes', 'y']
    # no = ['no', 'n']
    # while proceed not in yes and proceed not in no:
    #     proceed = input('Proceed ([y]/n)? ').lower()
    #     if proceed in no:
    #         print('Aborting...')
    #         return

    writer = SummaryWriter(
        f'{save_path}/{data_type}/{optim_name}_{loss_name}/{model_name}_{activation_function}/tensorboard/'
    )


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.device(device)


    ### load dataset ###
    N = 32
    data_gen = getattr(generate_simple_dataset, data_type)(N, save_path, train_fraction)
    train_loader, test_loader = data_gen.generate_dataset(data_size=data_size, batch_size=batch_size, noise=False, seed=random_seed, device=device)


    ### load model ###
    generator = getattr(deterministic_models, model_name)(data_gen.n_features, N, activation_function)

    # example_data, _ = next(iter(test_loader))
    # writer.add_graph(generator, example_data.cpu())

    # generator = torch.compile(generator) # supposed to improve training time (pytorch 2.0 feature)

    # use a pretrained model
    max_id = -1
    model_path = ''
    for root, dirs, files in os.walk(save_path):
        for name in files:
            if name.startswith(f'{model_name}_{activation_function}'):
                _id = int(name.split('_')[-1][:-3])
                if max_id < _id:
                    max_id = _id

                    if pre_trained_id == 'max':
                        model_path = os.path.join(root, name)

                    elif int(pre_trained_id) == _id:
                        model_path = os.path.join(root, name)
                
    if model_path != '' and pre_trained:
        print(f'\nUsing pretrained model at location: {model_path}\n''')
        generator.load_state_dict(torch.load(model_path))


    ### set optimizer and loss ###
    optimizer = getattr(optim, optim_name)(generator.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    loss_fn = getattr(F, loss_name)


    ### training loop ###
    generator.to(device)
    generator.train()

    train_steps = len(train_loader)
    test_steps = len(test_loader)

    for epoch in range(epochs):
        print(f'\nEPOCH {epoch + 1}:')
        train_loss = 0.
        start = time.time()
        for i, (X, target) in enumerate(train_loader):
            optimizer.zero_grad()

            pred = generator(X)

            loss = loss_fn(pred, target)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

            if i % 100 == 0 and i != 0:
                end = time.time()
                batch_speed = 100 / (end - start)
                progress = int((i / train_steps) * 20)
                bar = "\u2588" * progress + '-' * (20 - progress)
                print(f'Training: |{bar}| {5 * progress}% - loss {train_loss / 100:.2g} - speed {batch_speed:.2f} batch/s')
                writer.add_scalar('training loss', train_loss / 100, epoch * train_steps + i)
                train_loss = 0.
                start = time.time()

            if i % 1000 == 0 and i != 0:
                test_loss = 0.
                with torch.no_grad(): # evaluate model on test data
                    for _, (X, target) in enumerate(test_loader):
                        pred = generator(X)

                        loss = loss_fn(pred, target)
                        test_loss += loss.item()

                    writer.add_scalar('testing loss', test_loss / test_steps, epoch * train_steps + i)
                    print(f'\nValidation: loss {test_loss / test_steps:.2g}\n')

        scheduler.step()
                
        if epoch % save_interval == 0:
            torch.save(
                generator.state_dict(),
                f'{save_path}/{data_type}/{optim_name}_{loss_name}/{model_name}_{activation_function}/{model_name}_{activation_function}_{max_id+1}.pt'
            )



if __name__ == "__main__" :

    parser = argparse.ArgumentParser(description='Train different generative architectures on simplistic rings.', argument_default=argparse.SUPPRESS)
    parser.add_argument('-t', '--data_type', type=str, help="Type of data")
    parser.add_argument('-n', '--model_name', type=str, help="Name of the model")
    parser.add_argument('-a', '--activation_function', type=str, help="Activation function")
    parser.add_argument('--pre_trained', type=bool, action=argparse.BooleanOptionalAction, help="If we use a pretrained model")
    parser.add_argument('-id', '--pre_trained_id', type=str, help="If you want a specific pre-trained model")
    parser.add_argument('-s', '--save_path', type=str, help="Path where all the data is saved")
    parser.add_argument('-d', '--data_size', type=int, help="Number of events to train on")
    parser.add_argument('-e', '--epochs', type=int, help="Number of epochs to train for")
    parser.add_argument('-b', '--batch_size', type=int, help="Batch size")
    parser.add_argument('-o', '--optim_name', type=str, help="Name of the optimizer")
    parser.add_argument('-l', '--loss_name', type=str, help="Name of the loss function")
    parser.add_argument('-lr', '--lr', type=float, help="Learning rate")
    parser.add_argument('-f', '--train_fraction', type=float, help="Fraction of data used for training")
    parser.add_argument('-i', '--save_interval', type=int, help="Save network state every <save_interval> iterations")
    parser.add_argument('-r', '--random_seed', type=int, help="Random seed")

    args = parser.parse_args()

    train_model(**vars(args))

    # python3 training/train_bayesian_model.py -t energy_random -n DGBaNR -e 200 -d 64000 -b 64 -o Adam -l mse_loss -mc 20
