import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from mmdet.models.detectors.base import BaseDetector
from mmdet.models.detectors.test_mixins import RPNTestMixin, BBoxTestMixin, MaskTestMixin
from mmdet.models import builder
from mmdet.models.registry import DETECTORS
from mmdet.core import bbox2roi, bbox2result, build_assigner, build_sampler

from .seg_decoder import DeeplabDecoder
from .cap_decoder import CapDecoder

@DETECTORS.register_module
class EncoderDecoder(BaseDetector, RPNTestMixin, BBoxTestMixin,
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
                 seg_scales=[0.5, 0.75],
                 ):
        super(EncoderDecoder, self).__init__()
        self.backbone = builder.build_backbone(backbone)

        if neck is not None:
            self.neck = builder.build_neck(neck)

        if shared_head is not None:
            self.shared_head = builder.build_shared_head(shared_head)

        if rpn_head is not None:
            self.rpn_head = builder.build_head(rpn_head)

        if bbox_head is not None:
            self.bbox_roi_extractor = builder.build_roi_extractor(
                bbox_roi_extractor)
            self.bbox_head = builder.build_head(bbox_head)

        if mask_head is not None:
            if mask_roi_extractor is not None:
                self.mask_roi_extractor = builder.build_roi_extractor(
                    mask_roi_extractor)
                self.share_roi_extractor = False
            else:
                self.share_roi_extractor = True
                self.mask_roi_extractor = self.bbox_roi_extractor
            self.mask_head = builder.build_head(mask_head)

        if seg_cfg is not None:
            self.seg_decoder = DeeplabDecoder(**seg_cfg)
            self.seg_scales = seg_scales
        if cap_cfg is not None:
            self.cap_decoder = CapDecoder(**cap_cfg)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None
    
    @property
    def with_seg(self):
        return hasattr(self, 'seg_decoder') and self.seg_decoder is not None

    @property
    def with_cap(self):
        return hasattr(self, 'cap_decoder') and self.cap_decoder is not None

    def init_weights(self, pretrained=None):
        super(EncoderDecoder, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_bbox:
            self.bbox_roi_extractor.init_weights()
            self.bbox_head.init_weights()
        if self.with_mask:
            self.mask_head.init_weights()
            if not self.share_roi_extractor:
                self.mask_roi_extractor.init_weights()
        if self.with_seg:
            self.seg_decoder.init_weights()
        if self.with_cap:
            self.cap_decoder.init_weights()

    def extract_feat(self, img):
        # print(img[0].shape)
        # print(img[0][0])
        x = self.backbone(img)
        # print(x[0].shape)
        # print(x[0][0])
        if self.with_neck:
            # assert False, "shouldn't go inside"
            x = self.neck(x)
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
        x = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta,
                                          self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)

            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            proposal_inputs = rpn_outs + (img_meta, proposal_cfg)
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals

        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
            bbox_sampler = build_sampler(
                self.train_cfg.rcnn.sampler, context=self)
            num_imgs = img.size(0)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        # bbox head forward and loss
        if self.with_bbox:
            rois = bbox2roi([res.bboxes for res in sampling_results])
            # TODO: a more flexible way to decide which feature maps to use
            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            cls_score, bbox_pred = self.bbox_head(bbox_feats)

            bbox_targets = self.bbox_head.get_target(
                sampling_results, gt_bboxes, gt_labels, self.train_cfg.rcnn)
            loss_bbox = self.bbox_head.loss(cls_score, bbox_pred,
                                            *bbox_targets)
            losses.update(loss_bbox)

        # mask head forward and loss
        if self.with_mask:
            if not self.share_roi_extractor:
                pos_rois = bbox2roi(
                    [res.pos_bboxes for res in sampling_results])
                mask_feats = self.mask_roi_extractor(
                    x[:self.mask_roi_extractor.num_inputs], pos_rois)
                if self.with_shared_head:
                    mask_feats = self.shared_head(mask_feats)
            else:
                pos_inds = []
                device = bbox_feats.device
                for res in sampling_results:
                    pos_inds.append(
                        torch.ones(
                            res.pos_bboxes.shape[0],
                            device=device,
                            dtype=torch.uint8))
                    pos_inds.append(
                        torch.zeros(
                            res.neg_bboxes.shape[0],
                            device=device,
                            dtype=torch.uint8))
                pos_inds = torch.cat(pos_inds)
                mask_feats = bbox_feats[pos_inds]
            mask_pred = self.mask_head(mask_feats)

            mask_targets = self.mask_head.get_target(
                sampling_results, gt_masks, self.train_cfg.rcnn)
            pos_labels = torch.cat(
                [res.pos_gt_labels for res in sampling_results])
            loss_mask = self.mask_head.loss(mask_pred, mask_targets,
                                            pos_labels)
            losses.update(loss_mask)

        # seg head forward and loss
        if self.with_seg:
            logits = self.seg_decoder(x)
            if self.seg_scales is not None:
                _, _, H, W = logits.shape
                interp = lambda l: F.interpolate(l, size=(H, W), mode='bilinear', align_corners=False)
                logits_pyramid = []
                for p in self.seg_scales:
                    h = F.interpolate(img, scale_factor=p, mode='bilinear', align_corners=False)
                    logits_pyramid.append(self.seg_decoder(self.extract_feat(h)))
                logits_all = [logits] + [interp(l) for l in logits_pyramid]
                logits_max = torch.max(torch.stack(logits_all), dim=0)[0]
                logits_list = [logits] + logits_pyramid + [logits_max]
            else:
                logits_list = logits
            loss_seg = self.seg_decoder.loss(logits_list, torch.squeeze(gt_seg, dim=1), weight=self.seg_decoder.weight)
            losses.update(loss_seg)

        # cap head forward and loss
        if self.with_cap:
            predictions, caps_sorted, decode_lengths, alphas, sort_ind = self.cap_decoder(x[-1], gt_caps, gt_caplens)
            loss_cap = self.cap_decoder.loss(predictions, caps_sorted, decode_lengths, alphas, weight=self.cap_decoder.weight)
            losses.update(loss_cap)

        return losses

    # lixin
    def simple_test(self, img, img_meta, proposals=None, rescale=False,
            gt_seg=None, gt_caps=None, gt_caplens=None, allcaps=None):
        """Test without augmentation."""
        # assert self.with_bbox, "Bbox head must be implemented."

        x = self.extract_feat(img)

        result = {}
        
        if self.with_bbox:
            proposal_list = self.simple_test_rpn(
                x, img_meta, self.test_cfg.rpn) if proposals is None else proposals

            det_bboxes, det_labels = self.simple_test_bboxes(
                x, img_meta, proposal_list, self.test_cfg.rcnn, rescale=rescale)
            bbox_results = bbox2result(det_bboxes, det_labels,
                                       self.bbox_head.num_classes)

            if not self.with_mask:
                result['det'] = bbox_results
                # return bbox_results
            else:
                segm_results = self.simple_test_mask(
                    x, img_meta, det_bboxes, det_labels, rescale=rescale)
                result['det'] = (bbox_results, segm_results)
                # return bbox_results, segm_results

        if self.with_seg:
            logits = self.seg_decoder(x)
            # print(gt_seg.shape)
            _, _, H, W = gt_seg.shape
            logits = F.interpolate(logits, size=(H,W), mode='bilinear', align_corners=False)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            result['seg'] = {
                    'pred':preds.detach().cpu().numpy(),
                    'gt_seg':gt_seg.detach().cpu().numpy()
            }


        if self.with_cap:
            word_map_file = os.path.join("/mnt/coco17/annotations/caps", "WORDMAP_COCO_5_cap_per_img_5_min_word_freq.json")
            with open(word_map_file, 'r') as j:
                word_map = json.load(j)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = self.cap_decoder(x[-1], gt_caps, gt_caplens)
            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores_copy = scores.clone()
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)

            # References
            references = list()
            hypotheses = list()
            allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                        img_caps))  # remove <start> and pads
                references.append(img_captions)

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
            preds = temp_preds
            hypotheses.extend(preds)
            assert len(references) == len(hypotheses)
            # print(references)
            # print(hypotheses)
            # print(len(references))
            # print(len(hypotheses))
            # print(references[0])
            # print(hypotheses[0])
            result['cap'] = {
                # 'score':scores.detach().cpu().numpy(),
                # 'target':targets.detach().cpu().numpy(),
                'hypothesis':hypotheses[0],
                'reference':references[0]
            }


        return result

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
