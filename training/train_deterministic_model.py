import sys
sys.path.append('..')

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn

import matplotlib.pyplot as plt
import numpy as np

from DGBaN.datasets.generate_simple_dataset import ring_dataset, randomized_ring_dataset
from DGBaN.models import LinearGenerator, ConvGenerator

from IPython.display import clear_output



def train_model(
    data_type='Simple',
    model_name='Conv',
    model_path='',
    save_model='',
    data_size=10_000,
    epochs=10,
    batch_size=64,
    lr=1e-2,
    num_workers=8,
    train_fraction=0.8,
    save_interval=20,
    output_dir="",
    random_seed=42
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.device(device)


    # create a simplistic ring dataset
    N = 32
    if data_type == 'Simple':
        dataset_generator = ring_dataset(N=N)
    elif data_type == 'Random':
        dataset_generator = randomized_ring_dataset(N=N)

    train_loader, test_loader = dataset_generator.generate_dataset(data_size=data_size, batch_size=batch_size, seed=random_seed, device=device)
    train_loader
    test_loader


    if model_name == 'Linear':
        generator = LinearGenerator(dataset_generator.n_features, img_size=N).to(device)
    elif model_name == 'Conv':
        generator = ConvGenerator(dataset_generator.n_features, img_size=N).to(device)

    if model_path: # use a pretrained model
        generator.load_state_dict(torch.load(model_path))

    optimizer = optim.Adam(generator.parameters(), lr=lr)
    criterion = nn.MSELoss()


    train_loss = []
    test_loss = []
    for epoch in range(epochs):
        total_loss = 0
        for i, (features, real_images) in enumerate(train_loader):
            optimizer.zero_grad()

            generated_images = generator(features)
            loss = criterion(generated_images, real_images)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        train_loss.append(total_loss / batch_size)

        with torch.no_grad(): # evaluate model on test data
            total_loss = 0
            for i, (features, real_images) in enumerate(test_loader):
                generated_images = generator(features)
                loss = criterion(generated_images, real_images)
                total_loss += loss.item()
        
        test_loss.append(total_loss / batch_size)

        clear_output()
        print(f'EPOCH {epoch+1}: train loss - {train_loss[-1]:.2g}, tentest loss - {test_loss[-1]:.2g}')

        # plt.figure()
        # plt.plot(np.arange(len(train_loss)), train_loss, label='train')
        # plt.plot(np.arange(len(test_loss)), test_loss, label='test')
        # plt.legend()
        # plt.savefig('./figures/train_test.png')

        if epoch % 20 == 0 and epoch != 0:
            lr *= 1e-1
            optimizer = optim.Adam(generator.parameters(), lr=lr)
        
        if save_model and epoch % save_interval == 0 and epoch != 0:
            torch.save(generator.state_dict(), save_model)


    generator.eval()
    return generator, train_loss, test_loss, dataset_generator


if __name__ == "__main__" :

    parser = argparse.ArgumentParser(description='Train different generative architectures on simplistic rings.')
    parser.add_argument('-t', '--data_type', type=str, help="Type of data", default ="", required=False)
    parser.add_argument('-n', '--model_name', type=str, help="name of the model", default ="", required=False)
    parser.add_argument('-m', '--model_path', type=str, help="Pretained model path", default ="", required=False)
    parser.add_argument('-s', '--save_model', type=str, help="Pretained model path", default ="", required=False)
    parser.add_argument('-d', '--data_size', type=int, help="Number of events to train on", default=10_000, required=False)
    parser.add_argument('-e', '--epochs', type=int, help="Number of epochs to train for", default=10, required=False)
    parser.add_argument('-b', '--batch_size', type=int, help="Batch size", default=64, required=False)
    parser.add_argument('-l', '--lr', type=int, help="Learning rate", default=1e-2, required=False)
    parser.add_argument('-j', '--num_workers', type=int, help="Number of CPUs for loading data", default=8, required=False)
    parser.add_argument('-f', '--train_fraction', type=float, help="Fraction of data used for training", default=0.8, required=False)
    parser.add_argument('-i', '--save_interval', type=int, help="Save network state every <save_interval> iterations", default=20, required=False)
    parser.add_argument('-r', '--random_seed', type=int, help="Random seed", default=42, required=False)

    args = parser.parse_args()
    print(vars(args))
    
    train_model(**vars(args))

    # python3 train_model.py -t Random -n DGBaN -s ../save_model/first_DGBaN.pt -e 200 -d 10000 -b 64