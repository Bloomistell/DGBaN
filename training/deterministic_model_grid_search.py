import time

from collections import OrderedDict

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

from DGBaN import generate_simple_dataset, modular_DGBaN_base
import DGBaN.training.losses as losses
from DGBaN.models.deterministic_blocks import *
from DGBaN.utils import *


dict_blocks = {0:Conv3x3x3NormAct, 1:BottleNeck, 2:LinearBottleNeck, 3:Conv3x11x3NormAct}

block_choice = [np.random.choice([0, 1, 2, 3], size=3) for _ in range(20)]

models = []
for i in range(20):
    models.append({
        'linear_layers':LinearNormAct(6, 8000, 5),
        'unconv_1':dict_blocks[block_choice[i][0]](500, 250),
        'unconv_2':dict_blocks[block_choice[i][1]](250, 150),
        'unconv_3':dict_blocks[block_choice[i][2]](150, 75),
        'unconv_4':nn.Identity
    })



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
    train_loader, test_loader = data_gen.generate_dataset(data_size=data_size, batch_size=batch_size, noise=noise, seed=random_seed, device=device)


    ### set optimizer and loss ###
    optimizer = getattr(optim, optim_name)(generator.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_step, verbose=True)

    if loss_name in dir(losses):
        loss_fn = getattr(losses, loss_name)(N, device=device)
    
    elif loss_name in dir(F):
        loss_fn = getattr(F, loss_name)

    else:
        raise NameError(f'No loss named "{loss_name}" found.')


    ### Test models ###
    for model_name, model_dict in models.items():
        print(f"Testing {model_name}:")
        print("    Creating model...")
        generator = modular_DGBaN_base(N, model_dict)
        print("    SUCCESS\n")

        print(f"    Number of trainable parameters: {count_params(generator):,}")

        print("    Test training pahse...")
        for i, (X, target) in enumerate(test_loader):
            if i < 10:
                optimizer.zero_grad()

                pred = generator(X)

                loss = loss_fn(pred, target)

                train_loss += loss.item()

                loss.backward()
                optimizer.step()
        print("    SUCCESS\n")

    print("** Models valid **\n")

    print("Beginning grid search...")

    for model_name, model_dict in models.items():
        ### Starting time ###
        start_training = time.time()


        ### load model ###
        print(f"Training {model_name}:")
        generator = modular_DGBaN_base(N, model_dict)
 
        print(f'\nModel path: {save_path}/{data_type}/{optim_name}_{loss_name}/{model_name}/{model_name}_0.pt\n')

        writer = SummaryWriter(
            f'{save_path}/{data_type}/{optim_name}_{loss_name}/{model_name}/tensorboard_grid_search_{grid_search_id}/'
        )

        torch.save(
            generator.state_dict(),
            f'{save_path}/{data_type}/{optim_name}_{loss_name}/{model_name}/{model_name}_0.pt'
        )


        ### training loop ###
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

                if (i + scheduler_step) % step == 0 and i != 0:
                    scheduler.step()
                    scheduler_adjust = True                

                if i % 1000 == 0 and i != 0:
                    test_loss = 0.
                    with torch.no_grad(): # evaluate model on test data
                        for X, target in test_loader:
                            pred = generator(X)

                            loss = loss_fn(pred, target)
                            test_loss += loss.item()

                        writer.add_scalar('testing loss', test_loss / test_steps, epoch * train_steps + i)

                        len_adjust = len(" |{bar}| {2 * progress}% - ") - 2
                        print(f'Validation:{" " * len_adjust}loss {test_loss / test_steps:.2g}')
    
            if not scheduler_adjust:
                scheduler_step += train_steps

            if epoch % save_interval == 0:
                torch.save(
                    generator.state_dict(),
                    f'{save_path}/{data_type}/{optim_name}_{loss_name}/{model_name}/{model_name}_0.pt'
                )

            epoch += 1

        total_hours = (time.time() - start_training) / 3600
        total_minutes = (total_hours % 1) * 60

        print(f"""\n{model_name} TRAINING SUMMARY:
 - speed {batch_speed:.2f} batch/s
 - final train loss: {train_loss_scaled:.2g}
 - final test loss: {test_loss / test_steps / print_step:.2g}
 - total time: {total_hours:.0f}h {total_minutes:.0f}min
 - total epoch: {epoch}
 - epoch per hour: {epoch / total_hours} epoch/h
""")



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
    parser.add_argument('-r', '--random_seed', type=int, help="Random seed")

    parser.add_argument('-o', '--optim_name', type=str, help="Name of the optimizer")
    parser.add_argument('-l', '--loss_name', type=str, help="Name of the loss function")
    parser.add_argument('-lr', '--lr', type=float, help="Learning rate")
    parser.add_argument('-lrs', '--lr_step', type=float, help="Learning rate step")
    parser.add_argument('-stp', '--step', type=int, help="Step interval")

    args = parser.parse_args()

    train_model(**vars(args))

    # python3 training/train_bayesian_model.py -t energy_random -n DGBaNR -e 200 -d 64000 -b 64 -o Adam -l mse_loss -mc 20
