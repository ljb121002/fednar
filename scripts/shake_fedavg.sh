
CUDA_VISIBLE_DEVICES=$1 python main.py \
    --seed 0 \
    --algorithm "fedavg" \
    --dataset "shakespeare" \
    --model "shakespeare" \
    --num_clients 100 \
    --num_participating_clients 20 \
    --num_rounds 1000 \
    --alpha 0.3 \
    --cp 40 \
    --batch_size 100 \
    --weight_decay 1e-2 \
    --l2_reg 0 \
    --eta_l 0.1 \
    --eta_g 1.0 \
    --epsilon 0.0 \
    --mu 0.0 \
    --decay 0.995 \
    --max_norm 10.0 \
    --filename ./log/shake_fedavg_wd1e-2_l2reg0_clip_bs100cp40_decay0.995 \
    --use_gradient_clipping