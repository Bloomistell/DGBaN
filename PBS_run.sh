#!/bin/bash 
#PBS -S /bin/bash
#PBS -N train_DGBaN
#PBS -j oe
#PBS -o /home/J000000000007/DGBaN_project/run_logs/
#PBS -l select=1:ncpus=32:mem=32gb
#PBS -l walltime=10:00:00
#PBS -q mini

# activate env
source ~/.bashrc 
conda activate DGBaN

# go te working folder
cd /home/J000000000007/DGBaN_project/DGBaN/

# run python script
module load mpi/mpich-x86_64
mpirun -np 1 python training/train_bayesian_model.py -t pattern_random -n half_DGBaNR -a sigmoid -e 1000 -d 2000000 -b 32 -o Adam -l mse_loss -mc 5 -f 0.9 -i 1