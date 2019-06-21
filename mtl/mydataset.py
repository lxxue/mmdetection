import os.path as osp
import os

import mmcv
import numpy as np
from mmcv.parallel import DataContainer as DC
from torch.utils.data import Dataset

from mmdet.datasets.coco import CocoDataset

if __name__ == "__main__":
    img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    data_root = "/data/home/v-lixxue/coco17/"
    ann_file = data_root + 'annotations/instances_train2017.json'
    img_prefix = data_root + 'train2017/'
    img_scale = (1333, 800)
    dset = CocoDataset(ann_file, img_prefix, img_scale, img_norm_cfg)
    print(dset[0].keys())
    dset[0]['img']
