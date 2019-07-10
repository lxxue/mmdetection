# RANK=0 \
# CUDA_VISIBLE_DEVICES=4,5,6,7 \
#     python train.py \
#     mgpu_cap_r50_1x.py \
#     --validate \
#     --gpus 4 \
#     --seed 123 \
#     --launcher pytorch
CUDA_VISIBLE_DEVICES=4,5 \
    python -m torch.distributed.launch --nproc_per_node=2 \
    train.py \
    cap_r50_1x_stage2.py \
    --gpu 2 \
    --seed 123 \
    --launcher pytorch
