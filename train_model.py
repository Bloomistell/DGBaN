import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
import numpy as np

from generate_dataset import ring_dataset
from models import LinearGenerator, ConvGenerator

from IPython.display import clear_output



def train_model(model_name, model=None, data_size=10_000, epochs=10, batch_size=64, lr=1e-2, num_workers=8, train_fraction=0.8, save_interval=5000, output_dir="", random_seed=42):
    # create a simplistic ring dataset
    N = 32
    dataset_generator = ring_dataset(N=N)
    train_loader, test_loader = dataset_generator.generate_dataset(data_size=data_size, batch_size=batch_size, seed=random_seed)

    # Instantiate the generator and the optimizer
    if model is None:
        if model_name == 'Linear':
            generator = LinearGenerator(dataset_generator.n_features, img_size=N)
        elif model_name == 'Conv':
            generator = ConvGenerator(dataset_generator.n_features, img_size=N)
    else:
        generator = model
    optimizer = optim.Adam(generator.parameters(), lr=lr)

    # Loss function (Mean Squared Error)
    criterion = nn.MSELoss()

    # Training loop
    train_loss = []
    test_loss = []
    for epoch in range(epochs):
        total_loss = 0
        for i, (features, real_images) in enumerate(train_loader):
            # Zero the gradients
            optimizer.zero_grad()

            # Generate images from input features
            generated_images = generator(features)

            # Calculate the loss between generated images and target images
            loss = criterion(generated_images, real_images)
            total_loss += loss.item()

            # Backpropagate the gradients
            loss.backward()

            # Update the generator's parameters
            optimizer.step()

            # # Print progress
            # if i % 10 == 0:
            #     print(f"Epoch [{epoch+1}/{epochs}], Batch [{i+1}/{len(train_loader)}], Loss: {loss.item()}")

        train_loss.append(total_loss / batch_size)

        with torch.no_grad():
            total_loss = 0
            for i, (features, real_images) in enumerate(test_loader):
                generated_images = generator(features)
                loss = criterion(generated_images, real_images)
                total_loss += loss.item()
        
        test_loss.append(total_loss / batch_size)

        clear_output()
        print(f'EPOCH {epoch+1}...')
        plt.plot(np.arange(len(train_loss)), train_loss, label='train')
        plt.plot(np.arange(len(test_loss)), test_loss, label='test')
        plt.legend()
        plt.show()

        if epoch % 20 == 0 and epoch != 0:
            lr *= 1e-1
            optimizer = optim.Adam(generator.parameters(), lr=lr)
            print(lr)

    generator.eval()
    return generator, train_loss, test_loss, dataset_generator


if __name__ == "__main__" :

    parser = argparse.ArgumentParser(description='Train different generative architectures on simplistic rings.')
    parser.add_argument('-d', '--data_size', type=int, help="Number of events to train on", default=10_000, required=False)
    parser.add_argument('-e', '--epochs', type=int, help="Number of epochs to train for", default=10, required=False)
    parser.add_argument('-b', '--batch_size', type=int, help="Batch size", default=64, required=False)
    parser.add_argument('-l', '--lr', type=int, help="Learning rate", default=1e-2, required=False)
    parser.add_argument('-j', '--num_workers', type=int, help="Number of CPUs for loading data", default=8, required=False)
    parser.add_argument('-t', '--train_fraction', type=float, help="Fraction of data used for training", default=0.8, required=False)
    parser.add_argument('-s', '--save_interval', type=int, help="Save network state every <save_interval> iterations", default=5000, required=False)
    parser.add_argument('-o', '--output_dir', type=str, help="Output directory", default ="./", required=False)
    parser.add_argument('-r', '--random_seed', type=int, help="Random seed", default=42, required=False)

    args = parser.parse_args()
    print(vars(args))
    
    train_model(**vars(args))