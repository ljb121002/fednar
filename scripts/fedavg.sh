CUDA_VISIBLE_DEVICES=$1 python main.py \
    --seed 0 \
    --algorithm "fedavg" \
    --dataset "CIFAR10" \
    --model "resnet18" \
    --num_clients 100 \
    --num_participating_clients 20 \
    --num_rounds 1000 \
    --cp 20 \
    --alpha 0.3 \
    --weight_decay 0.05 \
    --eta_l 0.01 \
    --eta_g 1.0 \
    --epsilon 0.0 \
    --mu 0.0 \
    --filename ./log/fedavg \
    # --use_nar