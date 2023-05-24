CUDA_VISIBLE_DEVICES=$1 python main.py \
    --seed 0 \
    --algorithm "fedadam" \
    --dataset "CIFAR10" \
    --model "resnet18" \
    --num_clients 100 \
    --num_participating_clients 20 \
    --num_rounds 300 \
    --cp 20 \
    --alpha 0.3 \
    --weight_decay 0.01 \
    --eta_l 0.01 \
    --eta_g 0.1 \
    --epsilon 0.01 \
    --mu 0.0 \
    --filename ./log/fedadam \
    # --use_nar