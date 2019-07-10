CUDA_VISIBLE_DEVICES=4 \
    python train.py \
    cap_r50_1x_stage2.py \
    --validate \
    --gpus 1 \
    --seed 123
