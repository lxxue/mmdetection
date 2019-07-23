CUDA_VISIBLE_DEVICES=4,5,6,7 \
    # python -m torch.distributed.launch --nproc_per_node=4 \
    # test.py \
    python test.py \
    seg_r50_nofcn_2x.py \
    epoch_24.pth \
    --out results.pkl
