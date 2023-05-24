
CUDA_VISIBLE_DEVICES=$1 python main.py \
    --seed 0 \
    --algorithm "fedprox" \
    --dataset "shakespeare" \
    --model "shakespeare" \
    --num_clients 100 \
    --num_participating_clients 20 \
    --num_rounds 1000 \
    --alpha 0.3 \
    --cp 20 \
    --batch_size 500 \
    --weight_decay 1e-4 \
    --eta_l 0.1 \
    --eta_g 1.0 \
    --epsilon 0.0 \
    --mu 0.001 \
    --filename ./log/shake_fedprox \
    # --use_nar