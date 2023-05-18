import time

from collections import OrderedDict

import argparse

import pandas as pd

import torch
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

from datasets import generate_simple_dataset
from models import DGBase7Blocks
from training import losses
from models.deterministic_blocks import *
from utils import *



# dict_blocks = {0:BottleNeck, 1:Conv3x11x3NormAct, 2:Conv3x3x3NormAct}

# block_choice = []
# for i in range(3):
#     for j in range(3):
#         for k in range(3):
#             block_choice.append([i, j, k])

# models = {}
# for i in range(27):
#     models[f'model_{i+1}_sigmoid'] = {
#         'linear_layers':LinearNormAct(6, 215, 7),
#         'scale_up_1_channels':128,
#         'unconv_1':dict_blocks[block_choice[i][0]](128, 64),
#         'scale_up_2_channels':64,
#         'unconv_2':dict_blocks[block_choice[i][1]](64, 32),
#         'scale_up_3_channels':32,
#         'unconv_3':dict_blocks[block_choice[i][2]](32, 1, last_layer=True),
#         'sigmoid':True
#     }
#     models[f'model_{i+1}'] = {
#         'linear_layers':LinearNormAct(84, 8192, 4),
#         'scale_up_1_channels':128,
#         'unconv_1':dict_blocks[block_choice[i][0]](128, 64),
#         'scale_up_2_channels':64,
#         'unconv_2':dict_blocks[block_choice[i][1]](64, 32),
#         'scale_up_3_channels':32,
#         'unconv_3':dict_blocks[block_choice[i][2]](32, 1, last_layer=True),
#         'sigmoid':False
#     }


models = {'model_0':{
    'linear_layers':LinearNormAct(28, 1024, 5),
    'scale_up_1_channels':256,
    'unconv_1':BottleNeck(256, 128),
    'scale_up_2_channels':128,
    'unconv_2':Conv3x3x3NormAct(128, 64),
    'scale_up_3_channels':64,
    'unconv_3':Conv3x3x3NormAct(64, 32),
    'scale_up_4_channels':32,
    'unconv_4':Conv3x3x3NormAct(32, 16),
    'scale_up_5_channels':16,
    'unconv_5':Conv3x3x3NormAct(16, 8),
    'scale_up_6_channels':8,
    'unconv_6':Conv3x3x3NormAct(8, 4),
    'last_conv_channels':4
}}



def train_model(
    # Model:
    save_path='../save_data',
    save_interval=1,

    # Grid search:
    grid_search_id: int = 0,
    time_per_model: float = 1, # in hours

    # Data:
    data_type='energy_random',
    data_size=64000,
    epochs=200,
    batch_size=64,
    train_fraction=0.8,
    noise=False,
    features_degree=3,
    random_seed=42,

    # Training
    optim_name='Adam',
    loss_name='nll_loss',
    lr=1e-2,
    lr_step=0.95,
    step=1000
):
    print(
f"""
TRAINING SUMMARY:
    Grid search:
     - grid_search_id (-id): {grid_search_id}
     - time_per_model (-tpm): {time_per_model}

    Model:
     - save_path (-s): {save_path}
     - save_interval (-i): {save_interval}
    
    Data:
     - data_type (-t): {data_type}
     - data_size (-d): {data_size}
     - epochs (-e): {epochs}
     - batch_size (-b): {batch_size}
     - train_fraction (-f): {train_fraction}
     - noise (--noise): {noise}
     - random_seed (-r): {random_seed}

    Training:
     - optim_name (-o): {optim_name}
     - loss_name (-l): {loss_name}
     - lr (-lr): {lr}
     - lr_step (-lrs): {lr_step}
     - step (-stp): {step}
"""
    )
    time_per_model = time_per_model * 3600 # conversion in seconds

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.device(device)


    ### load dataset ###
    N = 32
    data_gen = getattr(generate_simple_dataset, data_type)(N, save_path, train_fraction)
    train_loader, test_loader = data_gen.generate_dataset(data_size=data_size, batch_size=batch_size, noise=noise, features_degree=features_degree, seed=random_seed, device=device)


    ### set loss ###
    if loss_name in dir(losses):
        loss_fn = getattr(losses, loss_name)(N, device=device)
    
    else: # if loss_name in dir(F):
        loss_fn = getattr(torch.nn.functional, loss_name)

    # else:
    #     raise NameError(f'No loss named "{loss_name}" found.')


    ### Test models ###
    for model_name, model_dict in models.items():
        print(f"Testing {model_name}:")
        print("    Model architecture:")
        for block_name, block in model_dict.items():
            print(f"{block_name}: {block.__name__ if isinstance(block, type) else block}")
        
        print("\n    Creating model...")
        generator = DGBase7Blocks(N, model_dict)
        print("    SUCCESS\n")

        print(f"    Number of trainable parameters: {count_params(generator):,}\n")

        print("    Test training phase...")

        optimizer = getattr(optim, optim_name)(generator.parameters(), lr=lr)

        generator.to(device)
        generator.train()
        for i, (X, target) in enumerate(test_loader):
            if i < 10:
                optimizer.zero_grad()

                pred = generator(X)

                loss = loss_fn(pred, target)

                loss.backward()
                optimizer.step()
        print("    SUCCESS\n")

    print("** Models valid **\n")

    print("Beginning grid search...")

    grid_search_dict = {"unconv_1": [], "unconv_2": [], "unconv_3": [], "num_parameters": [],
        "speed": [], "final_train_loss": [], "final_test_loss": [],
        "total_time": [], "total_epoch": [], "epoch_per_hour": []}
    for model_name, model_dict in models.items():
        ### Starting time ###
        start_training = time.time()

        grid_search_dict['unconv_1'].append(model_dict['unconv_1'].name)
        grid_search_dict['unconv_2'].append(model_dict['unconv_2'].name)
        grid_search_dict['unconv_3'].append(model_dict['unconv_3'].name)

        ### load model ###
        print(f"Training {model_name}:")
        generator = DGBase7Blocks(N, model_dict)

        grid_search_dict['num_parameters'].append(count_params(generator))
 
        print(f'\nModel path: {save_path}/{data_type}/{optim_name}_{loss_name}/grid_search_{grid_search_id}/{model_name}.pt\n')

        writer = SummaryWriter(
            f'{save_path}/{data_type}/{optim_name}_{loss_name}/grid_search_{grid_search_id}/tensorboard_{model_name}/'
        )

        torch.save(
            generator.state_dict(),
            f'{save_path}/{data_type}/{optim_name}_{loss_name}/grid_search_{grid_search_id}/{model_name}.pt'
        )


        ### training loop ###
        optimizer = getattr(optim, optim_name)(generator.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_step, verbose=True)

        generator.to(device)
        generator.train()

        train_steps = len(train_loader)
        test_steps = len(test_loader)

        print_step = train_steps / 50

        epoch = 0
        while time.time() - start_training < time_per_model:
            print(f'\nEPOCH {epoch + 1}:')
            train_loss = 0.

            scheduler_step = 0
            scheduler_adjust = False

            start = time.time()
            for i, (X, target) in enumerate(train_loader):
                optimizer.zero_grad()

                pred = generator(X)

                loss = loss_fn(pred, target)

                train_loss += loss.item()

                loss.backward()
                optimizer.step()

                if i % print_step == 0 and i != 0:
                    end = time.time()
                    
                    batch_speed = print_step / (end - start)
                    progress = int((i / train_steps) * 50)
                    bar = "\u2588" * progress + '-' * (50 - progress)
                    train_loss_scaled = train_loss / print_step
                    
                    print(f'Training: |{bar}| {2 * progress}% - loss {train_loss_scaled:.2g} - speed {batch_speed:.2f} batch/s')
                    
                    writer.add_scalar('training loss', train_loss / print_step, epoch * train_steps + i)

                    train_loss = 0.

                    start = time.time()

                if i % 1000 == 0 and i != 0:
                    test_loss = 0.
                    with torch.no_grad(): # evaluate model on test data
                        for X, target in test_loader:
                            pred = generator(X)

                            loss = loss_fn(pred, target)
                            test_loss += loss.item()

                        writer.add_scalar('testing loss', test_loss / test_steps, epoch * train_steps + i)

                        len_adjust = len(f" |{bar}| {2 * progress}% - ") - 2
                        print(f'Validation:{" " * len_adjust}loss {test_loss / test_steps:.2g}')
    
                # if (i + scheduler_step) % step == 0 and i != 0:
                #     scheduler.step()
                #     scheduler_adjust = True                

            if not scheduler_adjust:
                scheduler.step()

            if epoch % save_interval == 0:
                torch.save(
                    generator.state_dict(),
                    f'{save_path}/{data_type}/{optim_name}_{loss_name}/grid_search_{grid_search_id}/{model_name}.pt'
                )

            epoch += 1

        total_hours = (time.time() - start_training) / 3600
        total_minutes = (total_hours % 1) * 60

        print(f"""\n{model_name} TRAINING SUMMARY:
 - speed {batch_speed:.2f} batch/s
 - final train loss: {train_loss_scaled:.2g}
 - final test loss: {test_loss / test_steps:.2g}
 - total time: {int(total_hours)}h {total_minutes:.0f}min
 - total epoch: {epoch}
 - epoch per hour: {epoch / total_hours:.0f} epoch/h
""")
        
        grid_search_dict['speed'].append(batch_speed)
        grid_search_dict['final_train_loss'].append(train_loss_scaled)
        grid_search_dict['final_test_loss'].append(test_loss / test_steps)
        grid_search_dict['total_time'].append(f"{int(total_hours)}h {total_minutes:.0f}min")
        grid_search_dict['total_epoch'].append(epoch)
        grid_search_dict['epoch_per_hour'].append(epoch / total_hours)



if __name__ == "__main__" :

    parser = argparse.ArgumentParser(description='Train different generative architectures on simplistic rings.', argument_default=argparse.SUPPRESS)
    parser.add_argument('-id', '--grid_search_id', type=int, help="Save network state every <save_interval> iterations")
    parser.add_argument('-tpm', '--time_per_model', type=float, help="Save network state every <save_interval> iterations")

    parser.add_argument('-s', '--save_path', type=str, help="Path where all the data is saved")
    parser.add_argument('-i', '--save_interval', type=int, help="Save network state every <save_interval> iterations")

    parser.add_argument('-t', '--data_type', type=str, help="Type of data")
    parser.add_argument('-d', '--data_size', type=int, help="Number of events to train on")
    parser.add_argument('-e', '--epochs', type=int, help="Number of epochs to train for")
    parser.add_argument('-b', '--batch_size', type=int, help="Batch size")
    parser.add_argument('-f', '--train_fraction', type=float, help="Fraction of data used for training")
    parser.add_argument('--noise', type=bool, action=argparse.BooleanOptionalAction, help="Do we use a noised dataset")
    parser.add_argument('-fd', '--features_degree', type=int, help="Degree of the polynomial feature transformation")
    parser.add_argument('-r', '--random_seed', type=int, help="Random seed")

    parser.add_argument('-o', '--optim_name', type=str, help="Name of the optimizer")
    parser.add_argument('-l', '--loss_name', type=str, help="Name of the loss function")
    parser.add_argument('-lr', '--lr', type=float, help="Learning rate")
    parser.add_argument('-lrs', '--lr_step', type=float, help="Learning rate step")
    parser.add_argument('-stp', '--step', type=int, help="Step interval")

    args = parser.parse_args()

    train_model(**vars(args))

    # python3 training/train_bayesian_model.py -t energy_random -n DGBaNR -e 200 -d 64000 -b 64 -o Adam -l mse_loss -mc 20
