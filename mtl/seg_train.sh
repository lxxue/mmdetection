CUDA_VISIBLE_DEVICES=4 \
    python train.py \
    seg_r50_1x.py \
    --validate \
    --gpus 1 \
    --seed 123
