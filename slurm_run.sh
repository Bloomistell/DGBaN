#!/bin/sh
#SBATCH --job-name=DGBaN_training
#SBATCH --gres=gpu:1
#SBATCH --time=14:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH -o /home/J000000000007/DGBaN_project/grid_search_1.stdout

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=4

source ~/anaconda3/etc/profile.d/conda.sh
conda activate DGBaN

cd /home/J000000000007/DGBaN_project/DGBaN/
# python3 -u training/train_bayesian_model.py \
#  -n DGBaNR_2 \
#  --no-random_init \
#  -a sigmoid \
#  --no-use_base \
#  --no-vessel \
#  --pre_trained \
#  -id 2 \
#  -s ../save_data \
#  -i 1 \
# \
#  -t single_random_ring \
#  -d 640000 \
#  -e 200 \
#  -b 64 \
#  -f 0.8 \
#  --noise \
#  -sig 0.03 \
#  -r 42 \
# \
#  -o Adam \
#  -l mse_loss \
#  -mc 30 \
#  -lr 1e-4 \
#  -lrs 0.97 \
#  -stp 1000 \
#  --mean_training \
#  --no-std_training \

# deterministic:
# python3 -u training/train_deterministic_model.py \
#  -t single_random_ring \
#  -n DGBaNR_3_base \
#  -a sigmoid \
#  --no-pre_trained \
#  -id max \
#  -s ../save_data \
#  -i 1 \
# \
#  -d 640000 \
#  -e 200 \
#  -b 64 \
#  -f 0.8 \
#  --no-noise \
#  -r 42 \
# \
#  -o Adam \
#  -l mse_loss \
#  -lr 1e-2 \
#  -lrs 0.95 \
#  -stp 1000 \
 
# grid search:
# python3 -u training/deterministic_model_grid_search.py \
#  -id 0 \
#  -tpm 0.5 \
# \
#  -s ../save_data \
#  -i 1 \
# \
#  -t single_random_ring \
#  -d 640000 \
#  -e 200 \
#  -b 64 \
#  -f 0.8 \
#  --no-noise \
#  -r 42 \
# \
#  -o Adam \
#  -l mse_loss \
#  -lr 1e-2 \
#  -lrs 0.95 \
#  -stp 1000 \

# bayesian grid search:
python3 -u training/bayesian_model_grid_search.py \
 -id 0 \
 -tpm 2 \
\
 -s ../save_data \
 -i 1 \
\
 -t single_random_ring \
 -d 640000 \
 -e 200 \
 -b 64 \
 -f 0.8 \
 --noise \
 -sig 0.3 \
 -r 42 \
\
 -o Adam \
 -l mse_loss \
 -mc 20 \
 -lr 1e-2 \
 -lrs 0.95 \
 -stp 1000 \
 --mean_training \
 --no-std_training \