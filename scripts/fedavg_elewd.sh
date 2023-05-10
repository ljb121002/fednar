#!/bin/bash

#SBATCH --job-name=fedavg_cifar10_wd0.1                        # Job name
#SBATCH --output=./slurm_out/slurm_out.%A_%a.txt     # Standard output and error log
#SBATCH --nodes=1                                     # Run all processes on a single node   
#SBATCH --ntasks=1                                    # Run on a single CPU
#SBATCH --mem=40G                                 # Total RAM to be used
#SBATCH --cpus-per-task=32                     # Number of CPU cores
#SBATCH --gres=gpu:1                                # Number of GPUs (per node)
#SBATCH -p long                                          # Use the gpu partition
#SBATCH --time=24:00:00                          # Specify the time needed for your experiment

CUDA_VISIBLE_DEVICES=3 python main.py \
    --seed 0 \
    --algorithm "fedavg(elewd)" \
    --dataset "CIFAR10" \
    --model "resnet18" \
    --num_clients 100 \
    --num_participating_clients 20 \
    --num_rounds 2000 \
    --alpha 0.3 \
    --weight_decay 0.0 \
    --l2_reg 0.01 \
    --eta_l 0.1 \
    --eta_g 1.0 \
    --epsilon 0.0 \
    --mu 0.0 \
    --eps_elewd 1e-6 \
    --filename ./log/fedavg_elewd1e-6_lr0.1_reg0.01_clip \
    --use_gradient_clipping
    # --adjust_lr \
    # --adjustLR_coef 1.0 \