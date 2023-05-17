#!/bin/bash

#SBATCH --job-name=fedexp_cifar10_norm                        # Job name
#SBATCH --output=./slurm_out/slurm_out.%A_%a.txt     # Standard output and error log
#SBATCH --nodes=1                                     # Run all processes on a single node   
#SBATCH --ntasks=1                                    # Run on a single CPU
#SBATCH --mem=40G                                 # Total RAM to be used
#SBATCH --cpus-per-task=32                     # Number of CPU cores
#SBATCH --gres=gpu:1                                # Number of GPUs (per node)
#SBATCH -p long                                          # Use the gpu partition
#SBATCH --time=24:00:00                          # Specify the time needed for your experiment


# for wd in 1e-4 1e-3 1e-2 5e-2 1e-1
# for alpha in 1 10
# do
CUDA_VISIBLE_DEVICES=$1 python main.py \
        --seed 0 \
        --algorithm "fedexp" \
        --dataset "shakespeare" \
        --model "shakespeare" \
        --num_clients 100 \
        --num_participating_clients 20 \
        --num_rounds 1000 \
        --cp 100 \
        --batch_size 100 \
        --alpha 0.3 \
        --weight_decay 1e-3 \
        --l2_reg 0 \
        --eta_l 0.01 \
        --eta_g 'adaptive' \
        --epsilon 0.0 \
        --mu 0.0 \
        --decay 0.998 \
        --max_norm 10.0 \
        --filename ./log/shake_fedexp_lr0.01_wd1e-3_reg0_clip \
        --use_gradient_clipping
# done
