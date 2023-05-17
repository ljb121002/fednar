
CUDA_VISIBLE_DEVICES=$1 python main.py \
    --seed 0 \
    --algorithm "fedprox" \
    --dataset "CIFAR100" \
    --model "resnet34" \
    --num_clients 100 \
    --num_participating_clients 20 \
    --num_rounds 2000 \
    --cp 20 \
    --alpha 0.3 \
    --weight_decay 0.0 \
    --l2_reg 0.05 \
    --eta_l 0.01 \
    --eta_g 1.0 \
    --epsilon 0.0 \
    --mu 0.001 \
    --decay 0.998 \
    --max_norm 10.0 \
    --filename ./log/cifar100_fedprox_lr0.01_wd0.0_reg0.05_clip_res34 \
    --use_gradient_clipping