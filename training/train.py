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
    bayesian_models
)
import DGBaN.training.losses as losses



def get_model_paths(model_name, activation_function, save_path, data_type, optim_name, loss_name, bayesian, num_mc):
    max_id = -1
    start_name = f'{model_name}_{activation_function}'

    for _, _, files in os.walk(save_path):
        for name in files:
            if name.startswith(start_name) and name[-3:] != 'txt':
                _id = int(name.split('_')[-1][:-3])
                if max_id < _id:
                    max_id = _id

    if bayesian:
        tensorboard_path = f'{save_path}/{data_type}/{optim_name}_{loss_name}_{num_mc}/{model_name}_{activation_function}/tensorboard_{max_id+1}/'
        model_save_path = f'{save_path}/{data_type}/{optim_name}_{loss_name}_{num_mc}/{model_name}_{activation_function}/{model_name}_{activation_function}_{max_id+1}.pt'
        summary_path = f'{save_path}/{data_type}/{optim_name}_{loss_name}_{num_mc}/{model_name}_{activation_function}/{model_name}_{activation_function}_{max_id+1}.txt'
    else:
        tensorboard_path = f'{save_path}/{data_type}/{optim_name}_{loss_name}/{model_name}_{activation_function}/tensorboard_{max_id+1}/'
        model_save_path = f'{save_path}/{data_type}/{optim_name}_{loss_name}/{model_name}_{activation_function}/{model_name}_{activation_function}_{max_id+1}.pt'
        summary_path = f'{save_path}/{data_type}/{optim_name}_{loss_name}/{model_name}_{activation_function}/{model_name}_{activation_function}_{max_id+1}.txt'

    return max_id, model_save_path, tensorboard_path, summary_path



def load_pretrained(model_name, activation_function, max_id, pretrain_id, model, save_path):
    pretrain_path = ''
    if pretrain_id == 'max':
        pretrain_name = f'{model_name}_{activation_function}_{max_id}.pt'
    else:
        pretrain_name = f'{model_name}_{activation_function}_{pretrain_id}.pt'

    for root, dirs, files in os.walk(save_path):
        for name in files:
            if name == pretrain_name:
                pretrain_path = os.path.join(root, name)
                break
 
    if pretrain_path != '':
        print(f'\nUsing pretrained model at location: {pretrain_path}\n')
        model.load_state_dict(torch.load(pretrain_path))
    else:
        print('\nNo pretrained model found\n')



def load_base(activation_function, pretrain_id, model, save_path):
    max_base_id = -1
    base_path = ''
    base_name = f'{model.base_name}_{activation_function}'

    for root, dirs, files in os.walk(save_path):
        for name in files: 
            if name.startswith(base_name) and name[-3:] != 'txt':
                _id = int(name.split('_')[-1][:-3])
                if max_base_id < _id:
                    max_base_id = _id

                    if pretrain_id == 'max':
                        base_path = os.path.join(root, name)

                    elif int(pretrain_id) == _id:
                        base_path = os.path.join(root, name)
                
    weights = model.state_dict()
    pre_trained_weights = torch.load(base_path)

    for key in weights.keys():
        if key not in pre_trained_weights.keys():
            pre_trained_weights[key] = weights[key]
            
    print(f'\nUsing base model at location: {base_path}\n')
    model.load_state_dict(pre_trained_weights)



def load_vessel_base(activation_function, pretrain_id, model, save_path):
    max_base_id = -1
    base_path = ''
    base_name = f'{model.base_name}_{activation_function}'

    for root, dirs, files in os.walk(save_path):
        for name in files:        
            if name.startswith(base_name) and name[-3:] != 'txt':
                _id = int(name.split('_')[-1][:-3])
                if max_base_id < _id:
                    max_base_id = _id

                    if pretrain_id == 'max':
                        base_path = os.path.join(root, name)

                    elif int(pretrain_id) == _id:
                        base_path = os.path.join(root, name)
                
    weights = model.state_dict()
    pre_trained_weights = torch.load(base_path)

    for key_base, key_vessel in model.dict_dict_keys.items():
        weights[key_vessel] = pre_trained_weights[key_base]

    for name, param in model.named_parameters():
        if name in list(model.dict_dict_keys.values()):
            param.requires_grad = False

    print(f'\nUsing base model at location: {base_path}\n')
    model.load_state_dict(weights)



def fix_model_weights(model):
    for name, param in model.named_parameters():
        if 'weight' in name:
            param.requires_grad = False



class bayesian_training():
    def __init__(
        self,
        model,
        mean_training,
        std_training,
        pixel_training,
        sepixel_training,
        one_pixel_training,
        n_batch,
        kl_factor,
        num_mc,
        loss_fn,
        batch_size,
        optimizer
    ):
        self.model = model
        self.n_batch = n_batch
        self.kl_factor = kl_factor
        self.img_factor = 1
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        if mean_training:
            def inner_training(i, X, target, true_target=None):
                self.optimizer.zero_grad()

                preds = []
                kls = []
                for _ in range(num_mc): # extract several samples from the model
                    pred, kl = self.model(X)
                    preds.append(pred)
                    kls.append(kl)

                pred = torch.mean(torch.stack(preds), dim=0)
                kl = torch.mean(torch.stack(kls), dim=0)

                img = self.loss_fn(pred, target)
                loss = img * self.img_factor + kl * self.kl_factor

                self.train_loss += loss.item()
                self.self.img_loss += img.item()
                self.kl_loss += kl.item()

                loss.backward()
                self.optimizer.step()

        elif std_training:
            def inner_training(i, X, target, true_target=None):
                self.optimizer.zero_grad()

                pred, kl = self.model(X)

                img = self.loss_fn(pred, target)
                loss = img * self.img_factor + kl * self.kl_factor

                self.train_loss += loss.item()
                self.img_loss += img.item()
                self.kl_loss += kl.item()

                loss.backward()
                self.optimizer.step()

        elif pixel_training:
            def inner_training(i, X, target, true_target):
                pred, kl = self.model(X, true_target)

                pixel_loss, img, kl = self.loss_fn(pred, kl, target, self.img_factor, self.kl_factor)

                self.train_loss += pixel_loss.mean().item()
                self.img_loss += img.item()
                self.kl_loss += kl.item()

                for i in range(1024):
                    self.optimizer.zero_grad()
                    if i == 1023:
                        pixel_loss[i].backward()
                    else:
                        pixel_loss[i].backward(retain_graph=True)
                    self.optimizer.step()

        elif sepixel_training:
            def inner_training(i, X, target, true_target=None):
                pred, kl = self.model(X)
                img = self.loss_fn(pred, target)

                pixel_loss = img * self.img_factor + kl * self.kl_factor

                self.train_loss += pixel_loss.mean().item()
                self.img_loss += img.mean().item()
                self.kl_loss += kl.item()

                for i in range(1024):
                    self.optimizer.zero_grad()
                    if i == 1023:
                        pixel_loss[i].backward()
                    else:
                        pixel_loss[i].backward(retain_graph=True)
                    self.optimizer.step()

        elif one_pixel_training:
            def inner_training(i, X, target, true_target=None):
                self.optimizer.zero_grad()

                pred, kl = self.model(X)

                img = self.loss_fn(pred, target)
                pixel_loss = img * self.img_factor + kl * self.kl_factor

                self.train_loss += pixel_loss.item()
                self.img_loss += img.item()
                self.kl_loss += kl.item()

                pixel_loss.backward()
                self.optimizer.step()

        else:
            def inner_training(i, X, target, true_target=None):
                pred, kl = self.model(X)

                img = self.loss_fn(pred, target)
                loss = img * self.img_factor + kl * self.kl_factor

                self.train_loss += loss.item()
                self.img_loss += img.item()
                self.kl_loss += kl.item()
                
                loss.backward()

                if (i+1) % self.n_batch == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

        self.inner_training = inner_training


    def train(
        self,
        train_loader,
        epochs,
        pixel_training,
        batch_mean_training,
        kl_rate,
        adjust,
        scheduler,
        save_interval,
        model_save_path,
        writer
    ):
        train_steps = len(train_loader)
        print_step = train_steps // 50

        img_factor = 1

        for epoch in range(epochs):
            print(f'\nEPOCH {epoch + 1}:')
            self.train_loss = 0.
            self.img_loss = 0.
            self.kl_loss = 0.

            total_img_loss = 0.
            count = 0

            start = time.time()
            self.optimizer.zero_grad()
            for i, tensors in enumerate(train_loader):
                if pixel_training:
                    X, target, true_target = tensors
                    self.inner_training(i, X, target, true_target)
                else:
                    X, target = tensors
                    self.inner_training(i, X, target)

                if i % print_step == 0 and i != 0:
                    end = time.time()
                    
                    batch_speed = print_step / (end - start)
                    progress = int((i / train_steps) * 50)
                    bar = "\u2588" * progress + '-' * (50 - progress)
                    
                    print(f'Training: |{bar}| {2 * progress}% - loss {self.train_loss / print_step:.2g} - img {self.img_loss / print_step:.2g} - kl {self.kl_loss / print_step:.2g} - speed {batch_speed:.2f} batch/s')
                    
                    writer.add_scalar('training loss', self.train_loss / print_step, epoch * train_steps + i)
                    writer.add_scalar('img loss', self.img_loss / print_step, epoch * train_steps + i)
                    writer.add_scalar('kl loss', self.kl_loss / print_step, epoch * train_steps + i)

                    total_img_loss = self.img_loss / print_step
                    count += 1

                    self.train_loss = 0.
                    self.img_loss = 0.
                    self.kl_loss = 0.

                    start = time.time()

                # if i % 1000 == 0 and i != 0:
                #     test_loss = 0.
                #     with torch.no_grad(): # evaluate model on test data
                #         for i, (X, target) in enumerate(test_loader):
                #             pred, kl = model(X)

                #             loss = loss_fn(pred, target) + (kl / batch_size)
                #             test_loss += loss.item()

                #         writer.add_scalar('testing loss', test_loss / test_steps, epoch * train_steps + i)

                #         # getting the predictions for the base features
                #         pred_rings = np.zeros((n_samples, 32, 32))
                #         for i, feature in enumerate(features):
                #             pred_rings[i] += model(feature)[0].cpu().numpy().squeeze()

                #         pred_ring = pred_rings.sum(axis=0)
                #         pred_ring /= pred_ring.max()
                #         accuracy = 1 - ((true_ring - pred_ring)**2).mean()

                #         writer.add_scalar('accuracy', accuracy, epoch * train_steps + i)

                #         print(f'\nValidation: loss {test_loss / test_steps:.2g} - accuracy {accuracy:.2f}\n')

            if i % self.n_batch != 0 and batch_mean_training:  # Catch the case where the last mini-batch in the epoch is smaller than n_batch
                self.optimizer.step()
                self.optimizer.zero_grad()

            # if total_img_loss > adjust * 1.01:
            #     self.kl_factor /= kl_rate
            # else:
            #     self.kl_factor *= kl_rate

            # if total_img_loss / count > adjust * 1.1:
            #     self.kl_factor = min(self.img_factor, self.kl_factor)
            #     self.img_factor = 1
            # else:
            #     self.img_factor = min(self.img_factor, self.kl_factor)
            #     self.kl_factor = 1

            print()
            scheduler.step()

            if epoch % save_interval == 0:
                torch.save(
                    self.model.state_dict(),
                    model_save_path
                )

            print(f'Adjusting kl factor to {self.kl_factor:.4g}.')



def deterministic_training_loop(
        model,
        train_loader,
        test_loader,
        epochs,
        loss_fn,
        optimizer,
        scheduler,
        save_interval,
        model_save_path,
        writer
    ):

    train_steps = len(train_loader)
    test_steps = len(test_loader)

    print_step = train_steps / 50

    for epoch in range(epochs):
        print(f'\nEPOCH {epoch + 1}:')
        train_loss = 0.

        start = time.time()
        for i, (X, target) in enumerate(train_loader):
            optimizer.zero_grad()

            pred = model(X)

            loss = loss_fn(pred, target)

            train_loss += loss.item()

            loss.backward()
            optimizer.step()

            if i % print_step == 0 and i != 0:
                end = time.time()
                
                batch_speed = print_step / (end - start)
                progress = int((i / train_steps) * 50)
                bar = "\u2588" * progress + '-' * (50 - progress)
                
                print(f'Training: |{bar}| {2 * progress}% - loss {train_loss / print_step:.2g} - speed {batch_speed:.2f} batch/s')
                
                writer.add_scalar('training loss', train_loss / print_step, epoch * train_steps + i)

                train_loss = 0.

                start = time.time()

            if i % 1000 == 0 and i != 0:
                test_loss = 0.
                with torch.no_grad(): # evaluate model on test data
                    for X, target in test_loader:
                        pred = model(X)

                        loss = loss_fn(pred, target)
                        test_loss += loss.item()
                    
                    writer.add_scalar('testing loss', test_loss / test_steps, epoch * train_steps + i)

                    len_adjust = len(f" |{bar}| {2 * progress}% - ") - 2
                    print(f'Validation:{" " * len_adjust}loss {test_loss / test_steps:.2g}')

        scheduler.step()

        if epoch % save_interval == 0:
            torch.save(model.state_dict(), model_save_path)
