CUDA_VISIBLE_DEVICES=4,5,6,7 \
    python train.py \
    mgpu_cap_r50_1x.py \
    --gpus 4 \
    --seed 123 \
    --launcher pytorch
