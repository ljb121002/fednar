#!/bin/bash

#SBATCH --job-name=fedgc1_cifar10_wd1e-3_net_detach                        # Job name
#SBATCH --output=./slurm_out/slurm_out.%A_%a.txt     # Standard output and error log
#SBATCH --nodes=1                                     # Run all processes on a single node   
#SBATCH --ntasks=1                                    # Run on a single CPU
#SBATCH --mem=20G                                 # Total RAM to be used
#SBATCH --cpus-per-task=32                     # Number of CPU cores
#SBATCH --gres=gpu:1                                # Number of GPUs (per node)
#SBATCH -p long                                          # Use the gpu partition
#SBATCH --time=24:00:00                          # Specify the time needed for your experiment

CUDA_VISIBLE_DEVICES=3 python main.py \
    --seed 0 \
    --algorithm "fedavg(lrdd)" \
    --dataset "CIFAR10" \
    --model "resnet18" \
    --num_clients 100 \
    --num_participating_clients 20 \
    --num_rounds 2000 \
    --alpha 0.3 \
    --weight_decay 0.0 \
    --l2_reg 0.1 \
    --eta_l 0.01 \
    --eta_g 1.0 \
    --epsilon 0.0 \
    --mu 0.0 \
    --filename ./log/fedavg_lrdd_lr0.01_coef0_reg0.1_clip \
    --use_gradient_clipping \
    --adjust_lr \
    --adjustLR_coef 0.0 \