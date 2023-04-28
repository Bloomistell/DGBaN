#!/bin/sh
#SBATCH --job-name=DGBaN_training
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH -o /home/J000000000007/DGBaN_project/run2_log.stdout

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1

source ~/anaconda3/etc/profile.d/conda.sh
conda activate DGBaN

cd /home/J000000000007/DGBaN_project/DGBaN/
python3 -u training/train_bayesian_model.py \
 -t multi_random_ring \
 -n multi_b1conv_DGBaNR \
 -a sigmoid \
 --no-pre_trained \
 -e 1000 \
 -d 640000 \
 -b 32 \
 -o Adam \
 -l mse_loss \
 -mc 3 \
 -i 1