#!/bin/bash

#SBATCH --job-name=fedexp_cinic10                        # Job name
#SBATCH --output=./slurm_out/slurm_out.%A_%a.txt     # Standard output and error log
#SBATCH --nodes=1                                     # Run all processes on a single node   
#SBATCH --ntasks=1                                    # Run on a single CPU
#SBATCH --mem=40G                                 # Total RAM to be used
#SBATCH --cpus-per-task=32                     # Number of CPU cores
#SBATCH --gres=gpu:1                                # Number of GPUs (per node)
#SBATCH -p long                                          # Use the gpu partition
#SBATCH --time=24:00:00                          # Specify the time needed for your experiment

CUDA_VISIBLE_DEVICES=1 python main.py \
    --seed 0 \
    --algorithm "fedexp" \
    --dataset "CINIC10" \
    --model "resnet18" \
    --num_clients 200 \
    --num_participating_clients 20 \
    --num_rounds 2000 \
    --alpha 0.3 \
    --filename ./log/fedexp_cinic10_wd1e-4 \
    --weight_decay 0.0001 \
    --use_gradient_clipping