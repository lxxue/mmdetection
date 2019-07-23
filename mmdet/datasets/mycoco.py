import numpy as np
import os.path as osp
from pycocotools.coco import COCO
from collections import Counter
import json
import torch
from PIL import Image
from random import seed, choice, sample

from mmdet.datasets.coco import CocoDataset
import mmcv
from mmcv.parallel import DataContainer as DC

from mmdet.datasets.transforms import (ImageTransform, BboxTransform, MaskTransform,
                                       SegMapTransform, Numpy2Tensor)
from mmdet.datasets.utils import to_tensor, random_scale
from mmdet.datasets.extra_aug import ExtraAugmentation
from .registry import DATASETS

@DATASETS.register_module
class MyCocoDataset(CocoDataset):
    CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant',
               'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat',
               'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
               'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush')

    def __init__(self,
                 with_cap,
                 split,
                 cap_f,
                 cap_dir,
                 ann_file,
                 img_prefix,
                 img_scale,
                 img_norm_cfg,
                 multiscale_mode='value',
                 size_divisor=None,
                 proposal_file=None,
                 num_max_proposals=1000,
                 flip_ratio=0,
                 with_mask=True,
                 with_crowd=True,
                 with_label=True,
                 with_semantic_seg=False,
                 seg_prefix=None,
                 seg_scale_factor=1,
                 extra_aug=None,
                 resize_keep_ratio=True,
                 test_mode=False):
        super(MyCocoDataset, self).__init__(ann_file,
                                            img_prefix,
                                            img_scale,
                                            img_norm_cfg,
                                            multiscale_mode=multiscale_mode,
                                            size_divisor=size_divisor,
                                            proposal_file=proposal_file,
                                            num_max_proposals=num_max_proposals,
                                            flip_ratio=flip_ratio,
                                            with_mask=with_mask,
                                            with_crowd=with_crowd,
                                            with_label=with_label,
                                            with_semantic_seg=with_semantic_seg,
                                            seg_prefix=seg_prefix,
                                            seg_scale_factor=seg_scale_factor,
                                            extra_aug=extra_aug,
                                            resize_keep_ratio=resize_keep_ratio,
                                            test_mode=test_mode)
        # now load caption annotations
        min_word_freq = 5
        captions_per_image = 5
        max_len = 100

        self.cpi = captions_per_image
        self.with_cap = with_cap

        base_filename = 'COCO_' + str(captions_per_image) + '_cap_per_img_' + str(min_word_freq) + '_min_word_freq'
        captions_fname = osp.join(cap_dir, split + '_CAPTIONS_' + base_filename + '.json')
        caplens_fname = osp.join(cap_dir, split + '_CAPLENS_' + base_filename + '.json')
        word_map_fname = osp.join(cap_dir, 'WORDMAP_' + base_filename + '.json')

        # if cached
        if osp.isfile(word_map_fname) and osp.isfile(captions_fname) and osp.isfile(caplens_fname):
            with open(word_map_fname) as f:
                self.word_map = json.load(f)
            with open(captions_fname) as f:
                self.enc_captions = json.load(f)
            with open(caplens_fname) as f:
                self.caplens = json.load(f)
        else:
            self.split = split.lower()
            assert self.split in {'train', 'val'}
            with open(osp.join(cap_f), 'r') as f:
                data = json.load(f)
            img_caps = []
            word_freq = Counter()
            sorted_imgs = sorted(data['images'], key=lambda img : img['cocoid'])
            for img in sorted_imgs:
                captions = []
                for c in img['sentences']:
                    word_freq.update(c['tokens'])
                    if len(c['tokens']) <= max_len:
                        captions.append(c['tokens'])
                if img['split'] == self.split:
                    img_caps.append(captions)

            words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
            word_map = {k: v + 1 for v, k in enumerate(words)}
            word_map['<unk>'] = len(word_map) + 1
            word_map['<start>'] = len(word_map) + 1
            word_map['<end>'] = len(word_map) + 1
            word_map['<pad>'] = 0

            with open(word_map_fname, 'w') as f:
                json.dump(word_map, f)

            enc_captions = []
            caplens = []
            for cap in img_caps:
                if len(cap) < captions_per_image:
                    captions = cap + [choice(cap) for _ in range(captions_per_image-len(cap))]
                else:
                    captions = sample(cap, k=captions_per_image)
                assert len(captions) == captions_per_image

                for i, c in enumerate(captions):
                    enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + \
                            [word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))

                    # Find caption lengths
                    c_len = len(c) + 2
                    enc_captions.append(enc_c)
                    caplens.append(c_len)

            assert len(img_caps) * captions_per_image == len(enc_captions) == len(caplens)

            self.enc_captions = torch.LongTensor(enc_captions)
            self.caplens = torch.LongTensor(np.array(caplens).astype(np.long))
            # print(self.caplens[0])

            with open(captions_fname, 'w') as f:
                json.dump(enc_captions, f)

            with open(caplens_fname, 'w') as f:
                json.dump(caplens, f)

    def load_annotations(self, ann_file):
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.getCatIds()
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        # lixin
        self.img_ids = sorted(self.coco.getImgIds())
        img_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            img_infos.append(info)
        return img_infos

    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        # load image
        img = mmcv.imread(osp.join(self.img_prefix, img_info['filename']))
        # load proposals if necessary
        if self.proposals is not None:
            proposals = self.proposals[idx][:self.num_max_proposals]
            # TODO: Handle empty proposals properly. Currently images with
            # no proposals are just ignored, but they can be used for
            # training in concept.
            if len(proposals) == 0:
                return None
            if not (proposals.shape[1] == 4 or proposals.shape[1] == 5):
                raise AssertionError(
                    'proposals should have shapes (n, 4) or (n, 5), '
                    'but found {}'.format(proposals.shape))
            if proposals.shape[1] == 5:
                scores = proposals[:, 4, None]
                proposals = proposals[:, :4]
            else:
                scores = None

        ann = self.get_ann_info(idx)
        gt_bboxes = ann['bboxes']
        gt_labels = ann['labels']
        if self.with_crowd:
            gt_bboxes_ignore = ann['bboxes_ignore']

        # skip the image if there is no valid gt bbox
        if len(gt_bboxes) == 0:
            return None

        # extra augmentation
        if self.extra_aug is not None:
            img, gt_bboxes, gt_labels = self.extra_aug(img, gt_bboxes,
                                                       gt_labels)

        # apply transforms
        flip = True if np.random.rand() < self.flip_ratio else False
        # randomly sample a scale
        img_scale = random_scale(self.img_scales, self.multiscale_mode)
        img, img_shape, pad_shape, scale_factor = self.img_transform(
            img, img_scale, flip, keep_ratio=self.resize_keep_ratio)
        img = img.copy()
        if self.with_seg:
            gt_seg = mmcv.imread(
                osp.join(self.seg_prefix, img_info['file_name'].replace(
                    'jpg', 'png')),
                flag='unchanged')
            gt_seg = self.seg_transform(gt_seg.squeeze(), img_scale, flip)
            # gt_seg = mmcv.imrescale(
            #     gt_seg, self.seg_scale_factor, interpolation='nearest')
            gt_seg = resize_label(gt_seg, self.size_divisor) 
            gt_seg = gt_seg[None, ...]
        if self.proposals is not None:
            proposals = self.bbox_transform(proposals, img_shape, scale_factor,
                                            flip)
            proposals = np.hstack(
                [proposals, scores]) if scores is not None else proposals
        gt_bboxes = self.bbox_transform(gt_bboxes, img_shape, scale_factor,
                                        flip)
        if self.with_crowd:
            gt_bboxes_ignore = self.bbox_transform(gt_bboxes_ignore, img_shape,
                                                   scale_factor, flip)
        if self.with_mask:
            gt_masks = self.mask_transform(ann['masks'], pad_shape,
                                           scale_factor, flip)

        ori_shape = (img_info['height'], img_info['width'], 3)
        img_meta = dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=flip)

        data = dict(
            img=DC(to_tensor(img), stack=True),
            img_meta=DC(img_meta, cpu_only=True),
            gt_bboxes=DC(to_tensor(gt_bboxes)))
        if self.proposals is not None:
            data['proposals'] = DC(to_tensor(proposals))
        if self.with_label:
            data['gt_labels'] = DC(to_tensor(gt_labels))
        if self.with_crowd:
            data['gt_bboxes_ignore'] = DC(to_tensor(gt_bboxes_ignore))
        if self.with_mask:
            data['gt_masks'] = DC(gt_masks, cpu_only=True)
        if self.with_seg:
            data['gt_seg'] = DC(to_tensor(gt_seg.astype(np.long)), stack=True)
        if self.with_cap:
            rnd_idx = idx * self.cpi + np.random.randint(self.cpi)
            # data['gt_caps'] = DC(to_tensor(self.enc_captions[rnd_idx]))
            data['gt_caps'] = to_tensor(self.enc_captions[rnd_idx])
            # data['gt_caplens'] = DC(to_tensor(self.caplens[rnd_idx]))
            data['gt_caplens'] = to_tensor(np.array(self.caplens[rnd_idx]))
        return data

    def prepare_test_img(self, idx):
        """Prepare an image for testing (multi-scale and flipping)"""
        img_info = self.img_infos[idx]
        img = mmcv.imread(osp.join(self.img_prefix, img_info['filename']))
        if self.proposals is not None:
            proposal = self.proposals[idx][:self.num_max_proposals]
            if not (proposal.shape[1] == 4 or proposal.shape[1] == 5):
                raise AssertionError(
                    'proposals should have shapes (n, 4) or (n, 5), '
                    'but found {}'.format(proposal.shape))
        else:
            proposal = None

        def prepare_single(img, scale, flip, proposal=None):
            _img, img_shape, pad_shape, scale_factor = self.img_transform(
                img, scale, flip, keep_ratio=self.resize_keep_ratio)
            _img = to_tensor(_img)
            _img_meta = dict(
                ori_shape=(img_info['height'], img_info['width'], 3),
                img_shape=img_shape,
                pad_shape=pad_shape,
                scale_factor=scale_factor,
                flip=flip)
            if proposal is not None:
                if proposal.shape[1] == 5:
                    score = proposal[:, 4, None]
                    proposal = proposal[:, :4]
                else:
                    score = None
                _proposal = self.bbox_transform(proposal, img_shape,
                                                scale_factor, flip)
                _proposal = np.hstack(
                    [_proposal, score]) if score is not None else _proposal
                _proposal = to_tensor(_proposal)
            else:
                _proposal = None
            return _img, _img_meta, _proposal

        imgs = []
        img_metas = []
        proposals = []
        for scale in self.img_scales:
            _img, _img_meta, _proposal = prepare_single(
                img, scale, False, proposal)
            imgs.append(_img)
            img_metas.append(DC(_img_meta, cpu_only=True))
            proposals.append(_proposal)
            if self.flip_ratio > 0:
                _img, _img_meta, _proposal = prepare_single(
                    img, scale, True, proposal)
                imgs.append(_img)
                img_metas.append(DC(_img_meta, cpu_only=True))
                proposals.append(_proposal)
        data = dict(img=imgs, img_meta=img_metas)
        # if self.proposals is not None:
        #     data['proposals'] = proposals
        # if self.with_cap is not None:
        #     data['gt_caps'] = self.enc_captions[idx*self.cpi:(idx+1)*self.cpi]
        #     data['gt_caplens'] = self.caplens[idx*self.cpi:(idx+1)*self.cpi]
        if self.with_seg:
            gt_seg = mmcv.imread(
                osp.join(self.seg_prefix, img_info['file_name'].replace(
                    'jpg', 'png')),
                flag='unchanged')
            gt_seg = self.seg_transform(gt_seg.squeeze(), img_scale, flip)
            # gt_seg = mmcv.imrescale(
            #     gt_seg, self.seg_scale_factor, interpolation='nearest')
            gt_seg = resize_label(gt_seg, self.size_divisor) 
            gt_seg = gt_seg[None, ...]
            data['gt_seg'] = DC(to_tensor(gt_seg.astype(np.long)), stack=True)

        if self.with_cap:
            rnd_idx = idx * self.cpi + np.random.randint(self.cpi)
            # data['gt_caps'] = DC(to_tensor(self.enc_captions[rnd_idx]))
            data['gt_caps'] = to_tensor(self.enc_captions[rnd_idx])
            # data['gt_caplens'] = DC(to_tensor(self.caplens[rnd_idx]))
            data['gt_caplens'] = to_tensor(np.array(self.caplens[rnd_idx]))
            data['allcaps'] = to_tensor(self.enc_captions[idx*self.cpi:(idx+1)*self.api])
        return data

def resize_label(label, size_divisor):
    H, W = label.shape
    size = (W//size_divisor, H//size_divisor)
    label = Image.fromarray(label).resize(size, resample=Image.NEAREST)
    return np.asarray(label)

if __name__ == "__main__":
    img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    data_root = "/mnt/coco17/"
    ann_file = data_root + 'annotations/instances_train2017.json'
    img_prefix = data_root + 'train2017/'
    img_scale = (1333, 800)
    dset = MyCocoDataset(with_cap=True,
                         split='TRAIN',
                         cap_f=data_root+'annotations/caps_coco17.json',
                         cap_dir=data_root+'annotations/',
                         ann_file=ann_file,
                         img_prefix=img_prefix,
                         img_scale=img_scale,
                         img_norm_cfg=img_norm_cfg,
                         multiscale_mode='value',
                         size_divisor=None,
                         proposal_file=None,
                         num_max_proposals=1000,
                         flip_ratio=0,
                         with_mask=True,
                         with_crowd=True,
                         with_label=True,
                         with_semantic_seg=True,
                         seg_prefix=data_root+"annotations/train2017",
                         seg_scale_factor=1,
                         extra_aug=None,
                         resize_keep_ratio=True,
                         test_mode=False)
    print(dset[0])
    ann_file = data_root + 'annotations/instances_val2017.json'
    img_prefix = data_root + 'val2017/'
    dset = MyCocoDataset(with_cap=True,
                         split='VAL',
                         cap_f='/data/home/v-lixxue/coco17/annotations/caps_coco17.json',
                         cap_dir='/data/home/v-lixxue/coco17/annotations/',
                         ann_file=ann_file,
                         img_prefix=img_prefix,
                         img_scale=img_scale,
                         img_norm_cfg=img_norm_cfg,
                         multiscale_mode='value',
                         size_divisor=None,
                         proposal_file=None,
                         num_max_proposals=1000,
                         flip_ratio=0,
                         with_mask=True,
                         with_crowd=True,
                         with_label=True,
                         with_semantic_seg=True,
                         seg_prefix="/data/home/v-lixxue/coco17/annotations/val2017",
                         seg_scale_factor=1,
                         extra_aug=None,
                         resize_keep_ratio=True,
                         test_mode=True)
    print(dset[0])
