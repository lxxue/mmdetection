CUDA_VISIBLE_DEVICES=5,6,7 \
    python train.py \
    mtl_r50_caffe_c4_1x.py \
    --validate \
    --gpus 3 \
    --seed 123
