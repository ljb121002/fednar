for seed in 1 2
do
    CUDA_VISIBLE_DEVICES=$1 python main.py \
        --seed $seed \
        --algorithm "fedprox" \
        --dataset "CIFAR10" \
        --model "resnet18" \
        --num_clients 100 \
        --num_participating_clients 20 \
        --num_rounds 1000 \
        --cp 20 \
        --alpha 0.3 \
        --weight_decay 0.0 \
        --l2_reg 0.1 \
        --eta_l 0.01 \
        --eta_g 1.0 \
        --epsilon 0.0 \
        --mu 0.001 \
        --decay 0.998 \
        --max_norm 10.0 \
        --filename ./log/fedprox_lr0.01_wd0.0_reg0.1_clip_seed$seed \
        --use_gradient_clipping
done
