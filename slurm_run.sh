#!/bin/sh
#SBATCH --job-name=DGBaN_training
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH -o /home/J000000000007/DGBaN_project/run0_log.stdout

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

source ~/anaconda3/etc/profile.d/conda.sh
conda activate DGBaN

cd /home/J000000000007/DGBaN_project/DGBaN/
python3 -u training/train_bayesian_model.py \
 -t single_random_ring \
 -n bbuffer_DGBaNR \
 -a tanh \
 -e 1000 \
 -d 2000000 \
 -b 16 \
 -o Adam \
 -l mse_loss \
 -i 1 \
 --no-mean_training \
 -mc 0 \
 --no-pre_trained \
 --use_base \