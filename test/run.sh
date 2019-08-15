    # python test.py \
    python -m torch.distributed.launch --nproc_per_node=4 \
    --master_addr="127.0.0.1" --master_port=4321 \
    test.py \
    $1 \
    $2 \
    --out $3 \
    --launcher pytorch

