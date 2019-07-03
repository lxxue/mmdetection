CUDA_VISIBLE_DEVICES=6,7 \
    python train.py \
    cap_r50_1x_stage2.py \
    --validate \
    --gpus 2 \
    --seed 123
