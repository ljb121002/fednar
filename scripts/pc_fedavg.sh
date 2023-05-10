wd=$2

CUDA_VISIBLE_DEVICES=$1 python main.py \
    --seed 0 \
    --algorithm fedavg \
    --dataset personachat \
    --model gpt2 \
    --num_clients -1 \
    --num_participating_clients 1000 \
    --num_rounds 100 \
    --cp 10 \
    --alpha -1.0 \
    --weight_decay $wd \
    --l2_reg 0.0 \
    --eta_l 0.0001 \
    --eta_g 1.0 \
    --epsilon 0.0 \
    --mu 0.0 \
    --decay 0.998 \
    --max_norm 30.0 \
    --filename ./log/pc/fedavg_lr1e-4_wd${wd} \
    --use_gradient_clipping \
    # --use_debug_max \
    # --no_wandb \