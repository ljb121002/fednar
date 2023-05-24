CUDA_VISIBLE_DEVICES=$1 python main.py \
    --seed 0 \
    --algorithm "fedadam" \
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
    --eta_g 0.1 \
    --epsilon 0.01 \
    --mu 0.0 \
    --filename ./log/shake_fedadam \
    # --use_nar