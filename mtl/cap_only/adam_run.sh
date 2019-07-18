CUDA_VISIBLE_DEVICES=4,5,6,7 \
python -m torch.distributed.launch --nproc_per_node=4 \
    --master_addr="127.0.0.1" --master_port=8765 \
    ../train.py \
    cap_r50_nofcn_adam_2x.py \
    --gpu 4 \
    --seed 123 \
    --launcher pytorch
