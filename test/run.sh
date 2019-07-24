    # python test.py \
    python -m torch.distributed.launch --nproc_per_node=4 \
    --master_addr="127.0.0.1" --master_port=8765 \
    test.py \
    seg_r50_nofcn_2x.py \
    epoch_24.pth \
    --out 24_metric.pkl \
    --launcher pytorch

