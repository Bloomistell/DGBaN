#!/bin/sh
#SBATCH --job-name=DGBaN_training
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH -o /home/J000000000007/DGBaN_project/run_log.stdout

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

conda init bash
conda activate DGBaN
python3 training/train_bayesian_model.py -t multi_random -n multi_half_DGBaNR -a sigmoid -e 1000 -d 2000000 -b 32 -o Adam -l mse_loss -mc 3 -f 0.9 -i 1