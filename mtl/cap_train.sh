CUDA_VISIBLE_DEVICES=5 \
    python train.py \
    cap_r50_1x.py \
    --validate \
    --gpus 1 \
    --seed 123
