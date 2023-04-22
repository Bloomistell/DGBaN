#!/bin/sh
#SBATCH --job-name=DGBaN_training
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1

python3 training/train_bayesian_model.py -t energy_random -n big_DGBaNR -a relu -m ../save_model/big_DGBaNR_sigmoid_energy_random_1280000_32_Adam_mse_loss_5.pt -e 1000 -d 1280000 -b 32 -o Adam -l mse_loss -mc 5 -f 0.9 -i 1