CUDA_VISIBLE_DEVICES=0 \
    python train.py \
    mtl_r50_1x.py \
    --validate \
    --gpus 1 \
    --seed 123
