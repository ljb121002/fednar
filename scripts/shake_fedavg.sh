
CUDA_VISIBLE_DEVICES=$1 python main.py \
    --seed 0 \
    --algorithm "fedavg" \
    --dataset "shakespeare" \
    --model "shakespeare" \
    --num_clients 100 \
    --num_participating_clients 20 \
    --num_rounds 1000 \
    --alpha 0.3 \
    --cp 100 \
    --batch_size 100 \
    --weight_decay 1e-4 \
    --eta_l 0.1 \
    --eta_g 1.0 \
    --epsilon 0.0 \
    --mu 0.0 \
    --filename ./log/shake_fedavg \
    # --use_nar