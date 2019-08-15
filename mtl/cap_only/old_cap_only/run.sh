python -m torch.distributed.launch --nproc_per_node=8 \
    --master_addr="127.0.0.1" --master_port=8765 \
    ../train.py \
    cap_r50_nofcn_1x.py \
    --gpu 8 \
    --seed 123 \
    --launcher pytorch
