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
# python3 -u training/train_bayesian_model.py \
#  -t single_random_ring \
#  -n DGBaNR_2 \
#  --random_init \
#  -a tanh \
#  --no-use_base \
#  --no-vessel \
#  --no-pre_trained \
#  -id max \
#  -s ../save_data \
#  -i 1 \
# \
#  -d 640000 \
#  -e 200 \
#  -b 64 \
#  -f 0.8 \
#  --noise \
#  -sig 0.03 \
#  -r 42 \
# \
#  -o Adam \
#  -l LogL1Loss \
#  -mc 5 \
#  -lr 1e-2 \
#  -lrs 0.5 \
#  -stp 5000 \
#  --mean_training \
#  --no-std_training \

# deterministic:
python3 -u training/train_deterministic_model.py \
 -t single_random_ring \
 -n DGBaNR_2_base \
 -a tanh \
 --no-pre_trained \
 -id max \
 -s ../save_data \
 -i 1 \
\
 -d 640000 \
 -e 200 \
 -b 64 \
 -f 0.8 \
 -r 42 \
\
 -o Adam \
 -l LogL1Loss \
 -lr 1e-2 \