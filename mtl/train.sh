CUDA_VISIBLE_DEVICES=0 \
    python train.py \
    mtl_r50_1x.py \
    --work_dir ./ \
    --validate \
    --gpus 1 \
    --seed 123
