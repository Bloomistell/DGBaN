import time

from collections import OrderedDict

import argparse

import torch
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

from datasets import generate_simple_dataset
from models import DGBaN5Blocks
from training import losses
from models.bayesian_blocks import *
from utils import *


models = {
    f'model_1':{
        'linear_layers':LinearNormAct(6, 8192, 5),
        'scale_up_1_channels':512,
        'unconv_1':Conv3x3x3NormAct(512, 256),
        'scale_up_2_channels':256,
        'unconv_2':LinearBottleNeck(256, 128),
        'scale_up_3_channels':128,
        'unconv_3':Conv3x11x3NormAct(128, 64),
        'scale_up_4_channels':64,
        'unconv_4':BayesIdentity()
    },
    f'model_2':{
        'linear_layers':LinearNormAct(6, 8192, 5),
        'scale_up_1_channels':512,
        'unconv_1':LinearBottleNeck(512, 256),
        'scale_up_2_channels':256,
        'unconv_2':Conv3x3x3NormAct(256, 128),
        'scale_up_3_channels':128,
        'unconv_3':Conv3x11x3NormAct(128, 64),
        'scale_up_4_channels':64,
        'unconv_4':BayesIdentity()
    },
    f'model_3':{
        'linear_layers':LinearNormAct(6, 8192, 5),
        'scale_up_1_channels':512,
        'unconv_1':Conv3x11x3NormAct(512, 256),
        'scale_up_2_channels':256,
        'unconv_2':LinearBottleNeck(256, 128),
        'scale_up_3_channels':128,
        'unconv_3':Conv3x3x3NormAct(128, 64),
        'scale_up_4_channels':64,
        'unconv_4':BayesIdentity()
    },
    f'model_4':{
        'linear_layers':LinearNormAct(6, 8192, 5),
        'scale_up_1_channels':512,
        'unconv_1':Conv3x11x3NormAct(512, 256),
        'scale_up_2_channels':256,
        'unconv_2':Conv3x3x3NormAct(256, 128),
        'scale_up_3_channels':128,
        'unconv_3':LinearBottleNeck(128, 64),
        'scale_up_4_channels':64,
        'unconv_4':BayesIdentity()
    },
    f'model_5':{
        'linear_layers':LinearNormAct(6, 8192, 5),
        'scale_up_1_channels':512,
        'unconv_1':Conv3x3x3NormAct(512, 256),
        'scale_up_2_channels':256,
        'unconv_2':Conv3x11x3NormAct(256, 128),
        'scale_up_3_channels':128,
        'unconv_3':LinearBottleNeck(128, 64),
        'scale_up_4_channels':64,
        'unconv_4':BayesIdentity()
    },
    f'model_6':{
        'linear_layers':LinearNormAct(6, 8192, 5),
        'scale_up_1_channels':512,
        'unconv_1':LinearBottleNeck(512, 256),
        'scale_up_2_channels':256,
        'unconv_2':Conv3x11x3NormAct(256, 128),
        'scale_up_3_channels':128,
        'unconv_3':Conv3x3x3NormAct(128, 64),
        'scale_up_4_channels':64,
        'unconv_4':BayesIdentity()
    }
}


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
     - save_path (-s): {save_path}
     - save_interval (-i): {save_interval}
    
    Grid search:
     - grid_search_id (-id): {grid_search_id}
     - time_per_model (-tpm): {time_per_model}

    Data:
     - data_type (-t): {data_type}
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
    time_per_model = time_per_model * 3600 # conversion in seconds

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.device(device)


    ### load dataset ###
    N = 32
    data_gen = getattr(generate_simple_dataset, data_type)(N, save_path, train_fraction)
    train_loader, test_loader = data_gen.generate_dataset(data_size=data_size, batch_size=batch_size, noise=noise, seed=random_seed, device=device)


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
        generator = DGBaN5Blocks(N, model_dict)
        print("    SUCCESS\n")

        print(f"    Number of trainable parameters: {count_params(generator):,}\n")

        print("    Test training phase...")

        optimizer = getattr(optim, optim_name)(generator.parameters(), lr=lr)

        generator.to(device)
        generator.train()
        for i, (X, target) in enumerate(test_loader):
            if i < 10:
                optimizer.zero_grad()

                pred, kl = generator(X)

                loss = loss_fn(pred, target) + (kl / batch_size)

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
        generator = DGBaN5Blocks(N, model_dict)
 
        print(f'\nModel path: {save_path}/{data_type}/{optim_name}_{loss_name}/{model_name}/{model_name}_0.pt\n')

        writer = SummaryWriter(
            f'{save_path}/{data_type}/{optim_name}_{loss_name}/{model_name}/tensorboard_grid_search_{grid_search_id}/'
        )

        torch.save(
            generator.state_dict(),
            f'{save_path}/{data_type}/{optim_name}_{loss_name}/{model_name}/{model_name}_0.pt'
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
                    loss = img + (kl / batch_size)

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

                if i % print_step == 0 and i != 0:
                    end = time.time()
                    
                    batch_speed = print_step / (end - start)
                    progress = int((i / train_steps) * 50)
                    bar = "\u2588" * progress + '-' * (50 - progress)
                    train_loss_scaled = train_loss / print_step
                    img_loss_scaled = img_loss / print_step
                    kl_loss_scaled = kl_loss / print_step
                    
                    print(f'Training: |{bar}| {2 * progress}% - loss {train_loss_scaled:.2g} - img {img_loss_scaled:.2g} - kl {kl_loss_scaled:.2g} - speed {batch_speed:.2f} batch/s')
                    
                    writer.add_scalar('training loss', train_loss / print_step, epoch * train_steps + i)
                    writer.add_scalar('img loss', img_loss / print_step, epoch * train_steps + i)
                    writer.add_scalar('kl loss', kl_loss / print_step, epoch * train_steps + i)

                    train_loss = 0.

                    start = time.time()

                if (i + scheduler_step) % step == 0 and i != 0:
                    scheduler.step()
                    scheduler_adjust = True                

            if not scheduler_adjust:
                scheduler_step += train_steps

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
 - final img loss: {img_loss_scaled:.2g}
 - final kl loss: {kl_loss_scaled:.2g}
 - total time: {int(total_hours)}h {total_minutes:.0f}min
 - total epoch: {epoch}
 - epoch per hour: {epoch / total_hours:.0f} epoch/h
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

    train_model(**vars(args))

    # python3 training/train_bayesian_model.py -t energy_random -n DGBaNR -e 200 -d 64000 -b 64 -o Adam -l mse_loss -mc 20
