CUDA_VISIBLE_DEVICES=6,7 \
    python train.py \
    mgpu_cap_r50_1x.py \
    --validate \
    --gpus 2 \
    --seed 123
