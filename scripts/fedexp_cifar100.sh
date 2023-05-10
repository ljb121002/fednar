#!/bin/bash

#SBATCH --job-name=fedexp_cifar100                        # Job name
#SBATCH --output=./slurm_out/slurm_out.%A_%a.txt     # Standard output and error log
#SBATCH --nodes=1                                     # Run all processes on a single node   
#SBATCH --ntasks=1                                    # Run on a single CPU
#SBATCH --mem=40G                                 # Total RAM to be used
#SBATCH --cpus-per-task=32                     # Number of CPU cores
#SBATCH --gres=gpu:1                                # Number of GPUs (per node)
#SBATCH -p long                                          # Use the gpu partition
#SBATCH --time=24:00:00                          # Specify the time needed for your experiment

python main.py \
    --seed 0 \
    --algorithm "fedexp" \
    --dataset "CIFAR100" \
    --model "EMNIST_CNN" \
    --num_clients 100 \
    --num_participating_clients 20 \
    --num_rounds 2000 \
    --alpha 0.5 \
    --filename ./log/fedexp_alpha0.5