CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python train.py \
    mtl_r50_caffe_c4_1x.py \
    --validate \
    --gpus 8 \
    --seed 123
