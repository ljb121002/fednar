CUDA_VISIBLE_DEVICES=$1 python main.py \
        --seed 0 \
        --algorithm "fedexp" \
        --dataset "CIFAR10" \
        --model "resnet18" \
        --num_clients 100 \
        --num_participating_clients 20 \
        --num_rounds 1000 \
        --cp 20 \
        --alpha 0.3 \
        --weight_decay 0.01 \
        --eta_l 0.01 \
        --eta_g 'adaptive' \
        --epsilon 0.0 \
        --mu 0.0 \
        --filename ./log/fedexp \
        # --use_nar
