# RANK=0 \
# CUDA_VISIBLE_DEVICES=4,5,6,7 \
#     python train.py \
#     mgpu_cap_r50_1x.py \
#     --validate \
#     --gpus 4 \
#     --seed 123 \
#     --launcher pytorch
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python -m torch.distributed.launch --nproc_per_node=8 \
    --master_addr="127.0.0.1" --master_port=8765 \
    train.py \
    mtl_r50_caffe_c4_1x.py \
    --gpu 8 \
    --seed 123 \
    --launcher pytorch
