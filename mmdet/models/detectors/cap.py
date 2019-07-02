import torch
import torch.nn as nn

from PIL import Image
import numpy as np

from mmdet.models.detectors.base import BaseDetector
from mmdet.models.detectors.test_mixins import RPNTestMixin, BBoxTestMixin, MaskTestMixin
from mmdet.models import builder
from mmdet.models.registry import DETECTORS
from mmdet.core import bbox2roi, bbox2result, build_assigner, build_sampler

from .cap_decoder import CapDecoder

@DETECTORS.register_module
class CapModel(BaseDetector, RPNTestMixin, BBoxTestMixin,
                       MaskTestMixin):

    def __init__(self,
                 backbone,
                 neck=None,
                 shared_head=None,
                 rpn_head=None,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 seg_cfg=None,
                 cap_cfg=None,
                 ):
        super(CapModel, self).__init__()
        self.backbone = builder.build_backbone(backbone)

        # if neck is not None:
        #     self.neck = builder.build_neck(neck)

        # if shared_head is not None:
        #     self.shared_head = builder.build_shared_head(shared_head)

        # if rpn_head is not None:
        #     self.rpn_head = builder.build_head(rpn_head)

        # if bbox_head is not None:
        #     self.bbox_roi_extractor = builder.build_roi_extractor(
        #         bbox_roi_extractor)
        #     self.bbox_head = builder.build_head(bbox_head)

        # if mask_head is not None:
        #     if mask_roi_extractor is not None:
        #         self.mask_roi_extractor = builder.build_roi_extractor(
        #             mask_roi_extractor)
        #         self.share_roi_extractor = False
        #     else:
        #         self.share_roi_extractor = True
        #         self.mask_roi_extractor = self.bbox_roi_extractor
        #     self.mask_head = builder.build_head(mask_head)

        self.cap_decoder = CapDecoder(**cap_cfg)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    def init_weights(self, pretrained=None):
        super(CapModel, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        # if self.with_neck:
        #     if isinstance(self.neck, nn.Sequential):
        #         for m in self.neck:
        #             m.init_weights()
        #     else:
        #         self.neck.init_weights()
        # if self.with_shared_head:
        #     self.shared_head.init_weights(pretrained=pretrained)
        # if self.with_rpn:
        #     self.rpn_head.init_weights()
        # if self.with_bbox:
        #     self.bbox_roi_extractor.init_weights()
        #     self.bbox_head.init_weights()
        # if self.with_mask:
        #     self.mask_head.init_weights()
        #     if not self.share_roi_extractor:
        #         self.mask_roi_extractor.init_weights()
        self.cap_decoder.init_weights()
        # self.seg_decoder.init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        # for segmentation, we don't need fpn yet
        # but we will see what will happen if we took it
        # if self.with_neck:
        #     x = self.neck(x)
        return x


    def forward_train(self,
                      img,
                      img_meta,
                      gt_bboxes=None,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      gt_seg=None,
                      gt_caps=None,
                      gt_caplens=None):
        # print(img)
        # print(img.data)
        # print(img.data.shape)
        x = self.extract_feat(img)

        losses = dict()

        # cap head forward and loss
        predictions, caps_sorted, decode_lengths, alphas, sort_ind = self.cap_decoder(x[3], gt_caps, gt_caplens)
        loss_cap = self.cap_decoder.loss(predictions, caps_sorted, decode_lengths, alphas)
        losses.update(loss_cap)

        return losses

    def simple_test(self, img, img_meta, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."

        x = self.extract_feat(img)

        proposal_list = self.simple_test_rpn(
            x, img_meta, self.test_cfg.rpn) if proposals is None else proposals

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_meta, proposal_list, self.test_cfg.rcnn, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_meta, det_bboxes, det_labels, rescale=rescale)
            return bbox_results, segm_results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory
        proposal_list = self.aug_test_rpn(
            self.extract_feats(imgs), img_metas, self.test_cfg.rpn)
        det_bboxes, det_labels = self.aug_test_bboxes(
            self.extract_feats(imgs), img_metas, proposal_list,
            self.test_cfg.rcnn)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= img_metas[0][0]['scale_factor']
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(
                self.extract_feats(imgs), img_metas, det_bboxes, det_labels)
            return bbox_results, segm_results
        else:
            return bbox_results


# def resize_labels(labels, size):
#     # print(label.shape)
#     new_labels = []
#     for label in labels:
#         _, H, W = label.size()
#         label = label.detach().cpu().numpy().astype(np.uint8)
#     # print(label.dtype)
#     # print(Image.fromarray(label))
#         label = Image.fromarray(label).resize(size, resample=Image.NEAREST)
#         new_labels.append(np.asarray(label))
#     return torch.LongTensor(new_labels).cuda()

