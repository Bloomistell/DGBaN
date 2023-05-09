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

from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn

import numpy as np

from DGBaN import (
    generate_simple_dataset,
    bayesian_models
)
import losses



def train_bayes_model(
    # Model:
    data_type='energy_random',
    model_name='DGBaNR',
    random_init=False,
    activation_function='sigmoid',
    use_base=False,
    vessel=False,
    pre_trained=False,
    pre_trained_id='max',
    save_path='../save_data',
    save_interval=1,

    # Data:
    data_size=64000,
    epochs=200,
    batch_size=64,
    train_fraction=0.8,
    noise=True,
    sigma=0.3,
    random_seed=42,

    # Training
    optim_name='Adam',
    loss_name='nll_loss',
    num_mc=10,
    lr=1e-2,
    lr_step=0.1,
    step=9999,
    mean_training=True,
    std_training=True
):
    print(
f"""
TRAINING SUMMARY:
    Model:
     - data_type (-t): {data_type}
     - model_name (-n): {model_name}
     - random_init (-ri): {random_init}
     - activation_function (-a): {activation_function}
     - use_base (--use_base): {use_base}
     - vessel (--vessel): {vessel}
     - pre_trained (--pre_trained): {pre_trained}
     - pre_trained_id (-id): {pre_trained_id}
     - save_path (-s): {save_path}
     - save_interval (-i): {save_interval}
    
    Data:
     - data_size (-d): {data_size}
     - epochs (-e): {epochs}
     - batch_size (-b): {batch_size}
     - train_fraction (-f): {train_fraction}
     - noise (--noise): {noise}
     - sigma (-sig): {sigma}
     - random_seed (-r): {random_seed}

    Training:
     - optim_name (-o): {optim_name}
     - loss_name (-l): {loss_name}
     - num_mc (-mc): {num_mc}
     - lr (-lr): {lr}
     - lr_step (-lrs): {lr_step}
     - step (-stp): {step}
     - mean_training (--mean_training): {mean_training}
     - std_training (--std_training): {std_training}
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


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.device(device)


    ### load dataset ###
    N = 32
    data_gen = getattr(generate_simple_dataset, data_type)(N, save_path, train_fraction)
    train_loader, test_loader = data_gen.generate_dataset(data_size=data_size, batch_size=batch_size, noise=noise, sigma=sigma, seed=random_seed, device=device)


    ### load model ###
    generator = getattr(bayesian_models, model_name)(data_gen.n_features, N, activation_function, pre_trained_base=use_base)
    if random_init:
        generator.random_init()

    # use a pretrained model
    max_id = -1
    model_path = ''
    start_name = f'{model_name}_{activation_function}'

    max_base_id = -1
    base_path = ''
    base_name = f'{generator.base_name}_{activation_function}'

    for root, dirs, files in os.walk(save_path):
        for name in files:
            if name.startswith(start_name):
                _id = int(name.split('_')[-1][:-3])
                if max_id < _id:
                    max_id = _id

                    if pre_trained_id == 'max':
                        model_path = os.path.join(root, name)

                    elif int(pre_trained_id) == _id:
                        model_path = os.path.join(root, name)
                        
            if name.startswith(base_name):
                _id = int(name.split('_')[-1][:-3])
                if max_base_id < _id:
                    max_base_id = _id

                    if pre_trained_id == 'max':
                        base_path = os.path.join(root, name)

                    elif int(pre_trained_id) == _id:
                        base_path = os.path.join(root, name)
                
    if use_base:
        weights = generator.state_dict()
        pre_trained_weights = torch.load(base_path)

        for key in weights.keys():
            if key not in pre_trained_weights.keys():
                pre_trained_weights[key] = weights[key]
                
        print(f'\nUsing pretrained model at location: {base_path}\n')
        generator.load_state_dict(pre_trained_weights)

    elif vessel:
        weights = generator.state_dict()
        pre_trained_weights = torch.load(base_path)

        for key_base, key_vessel in generator.dict_dict_keys.items():
            weights[key_vessel] = pre_trained_weights[key_base]

        for name, param in generator.named_parameters():
            if name in list(generator.dict_dict_keys.values()):
                param.requires_grad = False

        print(f'\nUsing pretrained model at location: {base_path}\n')
        generator.load_state_dict(weights)

    elif model_path != '' and pre_trained:
        print(f'\nUsing pretrained model at location: {model_path}\n')
        generator.load_state_dict(torch.load(model_path))
        
    print(f'\nModel path: {save_path}/{data_type}/{optim_name}_{loss_name}_{num_mc}/{model_name}_{activation_function}/{model_name}_{activation_function}_{max_id+1}.pt\n')

    writer = SummaryWriter(
        f'{save_path}/{data_type}/{optim_name}_{loss_name}_{num_mc}/{model_name}_{activation_function}/tensorboard_{max_id+1}/'
    )

    torch.save(
        generator.state_dict(),
        f'{save_path}/{data_type}/{optim_name}_{loss_name}_{num_mc}/{model_name}_{activation_function}/{model_name}_{activation_function}_{max_id+1}.pt'
    )


    ### set optimizer and loss ###
    optimizer = getattr(optim, optim_name)(generator.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_step, verbose=True)

    if loss_name in dir(losses):
        loss_fn = getattr(losses, loss_name)(N, device=device)
    
    elif loss_name in dir(F):
        loss_fn = getattr(F, loss_name)

    else:
        raise NameError(f'No loss named "{loss_name}" found.')

    
    ### set accuracy variables
    # X, y = data_gen.generate_dataset(data_size=data_size, batch_size=batch_size, seed=random_seed, test_return=True)

    # n_samples = 1000
    # i = 0
    # while not X[i].any():
    #     i += 1

    # feature = X[i].reshape(1, -1)
    # features = [torch.Tensor(feature).to(device) for _ in range(n_samples)]
    # true_ring = data_gen.gaussian_from_features(*data_gen.scaler.inverse_transform(feature)[0].tolist())


    ### training loop ###
    generator.to(device)
    generator.train()

    train_steps = len(train_loader)
    test_steps = len(test_loader)

    print_step = train_steps / 50

    for epoch in range(epochs):
        print(f'\nEPOCH {epoch + 1}:')
        train_loss = 0.
        img_loss = 0.
        kl_loss = 0.

        scheduler_step = 0
        scheduler_adjust = False

        start = time.time()
        for i, (X, target) in enumerate(train_loader):
            if mean_training:
                optimizer.zero_grad()

                preds = []
                kls = []
                for _ in range(num_mc): # extract several samples from the model
                    pred, kl = generator(X)
                    preds.append(pred)
                    kls.append(kl)

                pred = torch.mean(torch.stack(preds), dim=0)
                kl = torch.mean(torch.stack(kls), dim=0)

                img = loss_fn(pred, target)
                loss = img # * (kl / batch_size)

                train_loss += loss.item()
                img_loss += img.item()
                kl_loss += (kl / batch_size).item()

                loss.backward()
                optimizer.step()

            if std_training:
                optimizer.zero_grad()

                pred, kl = generator(X)

                img = loss_fn(pred, target)
                loss = img + (kl / batch_size)

                train_loss += loss.item()
                img_loss += img.item()
                kl_loss += (kl / batch_size).item()

                loss.backward()
                optimizer.step()

            if i % 100 == 0 and i != 0:
                # img_factor, kl_factor = kl_factor, img_factor
                end = time.time()
                
                batch_speed = 100 / (end - start)
                progress = int((i / train_steps) * 50)
                bar = "\u2588" * progress + '-' * (50 - progress)
                
                print(f'Training: |{bar}| {2 * progress}% - loss {train_loss / (100 * 2):.2g} - img {img_loss / (100 * 2):.2g} - kl {kl_loss / (100 * 2):.2g} - speed {batch_speed:.2f} batch/s')
                
                writer.add_scalar('training loss', train_loss / (100 * 2), epoch * train_steps + i)
                writer.add_scalar('img loss', img_loss / (100 * 2), epoch * train_steps + i)
                writer.add_scalar('kl loss', kl_loss / (100 * 2), epoch * train_steps + i)

                train_loss = 0.
                img_loss = 0.
                kl_loss = 0.

                start = time.time()

            if (i + scheduler_step) % step == 0 and i != 0:
                print(f'\nLearning rate update:')
                scheduler.step()
                print('\n')
                scheduler_adjust = True                

            # if i % 1000 == 0 and i != 0:
            #     test_loss = 0.
            #     with torch.no_grad(): # evaluate model on test data
            #         for i, (X, target) in enumerate(test_loader):
            #             pred, kl = generator(X)

            #             loss = loss_fn(pred, target) + (kl / batch_size)
            #             test_loss += loss.item()

            #         writer.add_scalar('testing loss', test_loss / test_steps, epoch * train_steps + i)

            #         # getting the predictions for the base features
            #         pred_rings = np.zeros((n_samples, 32, 32))
            #         for i, feature in enumerate(features):
            #             pred_rings[i] += generator(feature)[0].cpu().numpy().squeeze()

            #         pred_ring = pred_rings.sum(axis=0)
            #         pred_ring /= pred_ring.max()
            #         accuracy = 1 - ((true_ring - pred_ring)**2).mean()

            #         writer.add_scalar('accuracy', accuracy, epoch * train_steps + i)

            #         print(f'\nValidation: loss {test_loss / test_steps:.2g} - accuracy {accuracy:.2f}\n')
  
        if not scheduler_adjust:
            scheduler_step += train_steps

        if epoch % save_interval == 0:
            torch.save(
                generator.state_dict(),
                f'{save_path}/{data_type}/{optim_name}_{loss_name}_{num_mc}/{model_name}_{activation_function}/{model_name}_{activation_function}_{max_id+1}.pt'
            )



if __name__ == "__main__" :

    parser = argparse.ArgumentParser(description='Train different generative architectures on simplistic rings.', argument_default=argparse.SUPPRESS)
    parser.add_argument('-t', '--data_type', type=str, help="Type of data")
    parser.add_argument('-n', '--model_name', type=str, help="Name of the model")
    parser.add_argument('--random_init', type=bool, action=argparse.BooleanOptionalAction, help="Initializing the gaussian's parameters at random")
    parser.add_argument('-a', '--activation_function', type=str, help="Activation function")
    parser.add_argument('--use_base', type=bool, action=argparse.BooleanOptionalAction, help="Using a deterministic base model for the deterministic layers")
    parser.add_argument('--vessel', type=bool, action=argparse.BooleanOptionalAction, help="Set only the weights of a bayesian model according to a base model")
    parser.add_argument('--pre_trained', type=bool, action=argparse.BooleanOptionalAction, help="If we use a pretrained model")
    parser.add_argument('-id', '--pre_trained_id', type=str, help="If you want a specific pre-trained model")
    parser.add_argument('-s', '--save_path', type=str, help="Path where all the data is saved")
    parser.add_argument('-i', '--save_interval', type=int, help="Save network state every <save_interval> iterations")

    parser.add_argument('-d', '--data_size', type=int, help="Number of events to train on")
    parser.add_argument('-e', '--epochs', type=int, help="Number of epochs to train for")
    parser.add_argument('-b', '--batch_size', type=int, help="Batch size")
    parser.add_argument('-f', '--train_fraction', type=float, help="Fraction of data used for training")
    parser.add_argument('--noise', type=bool, action=argparse.BooleanOptionalAction, help="Do we use a noised dataset")
    parser.add_argument('-sig', '--sigma', type=float, help="Sigma for the gaussian noise")
    parser.add_argument('-r', '--random_seed', type=int, help="Random seed")

    parser.add_argument('-o', '--optim_name', type=str, help="Name of the optimizer")
    parser.add_argument('-l', '--loss_name', type=str, help="Name of the loss function")
    parser.add_argument('-mc', '--num_mc', type=int, help="Number of Monte Carlo runs during training")
    parser.add_argument('-lr', '--lr', type=float, help="Learning rate")
    parser.add_argument('-lrs', '--lr_step', type=float, help="Learning rate step")
    parser.add_argument('-stp', '--step', type=int, help="Step interval")
    parser.add_argument('--mean_training', type=bool, action=argparse.BooleanOptionalAction, help="Train with the average of several mc")
    parser.add_argument('--std_training', type=bool, action=argparse.BooleanOptionalAction, help="Train with one mc")

    args = parser.parse_args()

    train_bayes_model(**vars(args))

    # python3 training/train_bayesian_model.py -t energy_random -n DGBaNR -e 200 -d 64000 -b 64 -o Adam -l mse_loss -mc 20
