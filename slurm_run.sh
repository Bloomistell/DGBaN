#!/bin/sh
#SBATCH --job-name=DGBaN_training
#SBATCH --gres=gpu:1
#SBATCH --time=40:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=2
#SBATCH -o /home/J000000000007/DGBaN_project/run1_log.stdout

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1

source ~/anaconda3/etc/profile.d/conda.sh
conda activate DGBaN

cd /home/J000000000007/DGBaN_project/DGBaN/
python3 -u training/train_model.py \
 -n OnePixel \
 --bayesian \
 --no-random_init \
 --no-fix_weights \
 -a no_activation_function \
 --no-use_base \
 --no-vessel \
 --no-pretrained \
 -id 11 \
 -s ../save_data \
 -i 1 \
\
 -t SinglePixel \
 -d 1000000 \
 -e 400 \
 -b 128 \
 -f 1 \
 --noise \
 -sig 0.3 \
 -fd 2 \
 -r 42 \
\
 -o Adam \
 -l mse_loss \
 -lt mse_loss \
 -kl 0.01 \
 -klr 1.1 \
 -mc 0 \
 -lr 1e-3 \
 -lrs 0.97 \
 --no-mean_training \
 --no-std_training \
 --no-pixel_training \
 --no-sepixel_training \
 --one_pixel_training \
 --no-batch_mean_training \
 -nb 8

# grid search:
# python3 -u training/deterministic_model_grid_search.py \
#  -id 8 \
#  -tpm 10 \
# \
#  -s ../save_data \
#  -i 1 \
# \
#  -t single_random_ring \
#  -d 640000 \
#  -e 200 \
#  -b 256 \
#  -f 0.8 \
#  --no-noise \
#  -fd 2 \
#  -r 42 \
# \
#  -o Adam \
#  -l mse_loss \
#  -lr 1e-2 \
#  -lrs 0.95 \
#  -stp 1000 \

# bayesian grid search:
# python3 -u training/bayesian_model_grid_search.py \
#  -id 0 \
#  -tpm 2 \
# \
#  -s ../save_data \
#  -i 1 \
# \
#  -t single_random_ring \
#  -d 640000 \
#  -e 200 \
#  -b 64 \
#  -f 0.8 \
#  --noise \
#  -sig 0.3 \
#  -fd 3 \
#  -r 42 \
# \
#  -o Adam \
#  -l mse_loss \
#  -mc 20 \
#  -lr 1e-2 \
#  -lrs 0.95 \
#  -stp 1000 \
#  --mean_training \
#  --no-std_training \