CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 \
    --master_addr="127.0.0.1" --master_port=8765 \
    ../train.py \
    configs/seg_r101_msc_poly_1x.py \
    --gpu 4 \
    --seed 123 \
    --launcher pytorch
