CUDA_VISIBLE_DEVICES=$1 python main.py \
    --seed 0 \
    --algorithm "fedavgm" \
    --dataset "shakespeare" \
    --model "shakespeare" \
    --num_clients 100 \
    --num_participating_clients 20 \
    --num_rounds 1000 \
    --cp 100 \
    --batch_size 100 \
    --alpha 0.3 \
    --weight_decay 1e-4 \
    --eta_l 0.01 \
    --eta_g 1.0 \
    --epsilon 0.0 \
    --mu 0.0 \
    --filename ./log/shake_fedavgm \
    # --use_nar