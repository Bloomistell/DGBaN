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
from torch.utils.tensorboard import SummaryWriter

import numpy as np

from DGBaN import (
    generate_simple_dataset,
    bayesian_models,
    deterministic_models
)
import DGBaN.training.losses as losses
from DGBaN.training.train import (
    bayesian_training_loop,
    deterministic_training_loop,
    get_model_paths,
    load_pretrained,
    load_base,
    load_vessel_base
)
from DGBaN.utils import *



if __name__ == "__main__" :

    parser = argparse.ArgumentParser(description='Train different generative architectures on simplistic rings.', argument_default=argparse.SUPPRESS)
    parser.add_argument('-n', '--model_name', type=str, help="Name of the model")
    parser.add_argument('--bayesian', type=bool, action=argparse.BooleanOptionalAction, help="Whether the model is bayesian or not")
    parser.add_argument('--random_init', type=bool, action=argparse.BooleanOptionalAction, help="Initializing the gaussian's parameters at random")
    parser.add_argument('-a', '--activation_function', type=str, help="Activation function")
    parser.add_argument('--use_base', type=bool, action=argparse.BooleanOptionalAction, help="Using a deterministic base model for the deterministic layers")
    parser.add_argument('--vessel', type=bool, action=argparse.BooleanOptionalAction, help="Set only the weights of a bayesian model according to a base model")
    parser.add_argument('--pretrained', type=bool, action=argparse.BooleanOptionalAction, help="If we use a pretrained model")
    parser.add_argument('-id', '--pretrained_id', type=str, help="If you want a specific pre-trained model")
    parser.add_argument('-s', '--save_path', type=str, help="Path where all the data is saved")
    parser.add_argument('-i', '--save_interval', type=int, help="Save network state every <save_interval> iterations")

    parser.add_argument('-t', '--data_type', type=str, help="Type of data")
    parser.add_argument('-d', '--data_size', type=int, help="Number of events to train on")
    parser.add_argument('-e', '--epochs', type=int, help="Number of epochs to train for")
    parser.add_argument('-b', '--batch_size', type=int, help="Batch size")
    parser.add_argument('-f', '--train_fraction', type=float, help="Fraction of data used for training")
    parser.add_argument('--noise', type=bool, action=argparse.BooleanOptionalAction, help="Do we use a noised dataset")
    parser.add_argument('-sig', '--sigma', type=float, help="Sigma for the gaussian noise")
    parser.add_argument('-fd', '--features_degree', type=int, help="Degree of the polynomial feature transformation")
    parser.add_argument('-r', '--random_seed', type=int, help="Random seed")

    parser.add_argument('-o', '--optim_name', type=str, help="Name of the optimizer")
    parser.add_argument('-l', '--loss_name', type=str, help="Name of the loss function")
    parser.add_argument('-lt', '--loss_type', type=str, help="Type of the loss function")
    parser.add_argument('-kl', '--kl_factor', type=float, help="Factor for kl in the loss function")
    parser.add_argument('-klr', '--kl_rate', type=float, help="Rare of increase of kl")
    parser.add_argument('-mc', '--num_mc', type=int, help="Number of Monte Carlo runs during training")
    parser.add_argument('-lr', '--lr', type=float, help="Learning rate")
    parser.add_argument('-lrs', '--lr_step', type=float, help="Learning rate step")
    parser.add_argument('--mean_training', type=bool, action=argparse.BooleanOptionalAction, help="Train with the average of several mc")
    parser.add_argument('--std_training', type=bool, action=argparse.BooleanOptionalAction, help="Train with one mc")

    args = parser.parse_args()

    # python3 -u training/train_bayesian_model.py -t energy_random -n DGBaNR -e 200 -d 64000 -b 64 -o Adam -l mse_loss -mc 20


    # Model:
    model_name = args.model_name                    # 'DGBaNR'
    bayesian = args.bayesian                        # True
    random_init = args.random_init                  # False
    activation_function = args.activation_function  # 'sigmoid'
    use_base = args.use_base                        # False
    vessel = args.vessel                            # False
    pretrained = args.pretrained                    # False
    pretrained_id = args.pretrained_id              # 'max'
    save_path = args.save_path                      # '../save_data'
    save_interval = args.save_interval              # 1

    # Data:
    data_type = args.data_type                      # 'energy_random'
    data_size = args.data_size                      # 64000
    epochs = args.epochs                            # 200
    batch_size = args.batch_size                    # 64
    train_fraction = args.train_fraction            # 0.8
    noise = args.noise                              # True
    sigma = args.sigma                              # 0.3
    features_degree = args.features_degree          # 1
    random_seed = args.random_seed                  # 42

    # Training
    optim_name = args.optim_name                    # 'Adam'
    loss_name = args.loss_name                      # 'nll_loss'
    loss_type = args.loss_type                      # 'mse_loss'
    kl_factor = args.kl_factor                      # 0.1
    kl_rate = args.kl_rate                        # 1.1
    num_mc = args.num_mc                            # 10
    lr = args.lr                                    # 1e-2
    lr_step = args.lr_step                          # 0.1
    mean_training = args.mean_training              # True
    std_training = args.std_training                # True
    

    summary = f"""TRAINING SUMMARY:
    Model:
     - model_name (-n): {model_name}
     - random_init (-ri): {random_init}
     - activation_function (-a): {activation_function}
     - use_base (--use_base): {use_base}
     - vessel (--vessel): {vessel}
     - pretrained (--pretrained): {pretrained}
     - pretrained_id (-id): {pretrained_id}
     - save_path (-s): {save_path}
     - save_interval (-i): {save_interval}
    
    Data:
     - data_type (-t): {data_type}
     - data_size (-d): {data_size}
     - epochs (-e): {epochs}
     - batch_size (-b): {batch_size}
     - train_fraction (-f): {train_fraction}
     - noise (--noise): {noise}
     - sigma (-sig): {sigma}
     - features_degree (-fd): {features_degree}
     - random_seed (-r): {random_seed}

    Training:
     - optim_name (-o): {optim_name}
     - loss_name (-l): {loss_name}
     - kl_factor (-kl): {kl_factor}
     - kl_rate (-kl): {kl_rate}
     - num_mc (-mc): {num_mc}
     - lr (-lr): {lr}
     - lr_step (-lrs): {lr_step}
     - mean_training (--mean_training): {mean_training}
     - std_training (--std_training): {std_training}
"""
    print(summary)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.device(device)


    ### load dataset ###
    N = 32
    data_gen = getattr(generate_simple_dataset, data_type)(N, save_path, train_fraction)
    train_loader, test_loader = data_gen.generate_dataset(
        data_size=data_size,
        batch_size=batch_size,
        noise=noise,
        sigma=sigma,
        features_degree=features_degree,
        seed=random_seed,
        device=device
    )

    adjust = data_gen.noise_delta


    ### load model ###
    if bayesian:
        generator = getattr(bayesian_models, model_name)(data_gen.n_features, N, activation_function, pretrained_base=use_base)
        if random_init:
            generator.random_init()
    else:
        generator = getattr(deterministic_models, model_name)(data_gen.n_features, N, activation_function)

    max_id, model_save_path, tensorboard_path, summary_path = get_model_paths(
        model_name,
        activation_function,
        save_path,
        data_type,
        optim_name,
        loss_name,
        bayesian,
        num_mc
    )

    print(generator)

    print(f'\nNumber of parameters: {count_params(generator):,}\n')


    ### load pretrained model ###
    if pretrained:
        load_pretrained(model_name, activation_function, max_id, pretrained_id, generator, save_path)

    elif use_base:
        load_base(activation_function, pretrained_id, generator, save_path)
        
    elif vessel:
        load_vessel_base(activation_function, pretrained_id, generator, save_path)


    ### set optimizer and loss ###
    optimizer = getattr(optim, optim_name)(generator.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_step, verbose=True)

    if loss_name in dir(losses):
        loss_fn = getattr(losses, loss_name)(N=N, device=device, adjust=data_gen.noise_delta.item()[loss_type]['mean'])
    
    elif loss_name in dir(torch.nn.functional):
        loss_fn = getattr(torch.nn.functional, loss_name)

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


    writer = SummaryWriter(tensorboard_path)

    print(f'\nModel path:\n{model_save_path}\n')
    torch.save(generator.state_dict(), model_save_path)
    save_txt(summary, summary_path)


    ### training loop ###
    generator.to(device)
    generator.train()

    if bayesian:
        bayesian_training_loop(
            generator,
            train_loader,
            epochs,
            mean_training,
            std_training,
            kl_factor,
            kl_rate,
            num_mc,
            loss_fn,
            adjust,
            batch_size,
            optimizer,
            scheduler,
            save_interval,
            model_save_path,
            writer
        )

    else:
        deterministic_training_loop(
            generator,
            train_loader,
            test_loader,
            epochs,
            loss_fn,
            optimizer,
            scheduler,
            save_interval,
            model_save_path,
            writer
        )

    print('*** Training finished ***')
