for alpha in 1 10
do
    CUDA_VISIBLE_DEVICES=$1 python main.py \
        --seed 0 \
        --algorithm "fedprox" \
        --dataset "CIFAR10" \
        --model "resnet18" \
        --num_clients 100 \
        --num_participating_clients 20 \
        --num_rounds 2000 \
        --cp 20 \
        --alpha $alpha \
        --weight_decay 0.001 \
        --l2_reg 0.0 \
        --eta_l 0.01 \
        --eta_g 1.0 \
        --epsilon 0.0 \
        --mu 0.001 \
        --decay 0.998 \
        --max_norm 10.0 \
        --filename ./log/fedprox_lr0.01_wd0.001_reg0.0_clip_alpha$alpha \
        --use_gradient_clipping
done
