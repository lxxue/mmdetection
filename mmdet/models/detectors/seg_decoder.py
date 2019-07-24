import torch
import torch.nn as nn
import numpy as np
from PIL import Image

from mmdet.models.builder import build_loss


class DeeplabDecoder(nn.Module):
    def __init__(self, ignore_label, inplanes, num_classes, atrous_rates, weight=1.0):
        super(DeeplabDecoder, self).__init__()
        self.aspp = _ASPP(inplanes, num_classes, atrous_rates)
        self.loss_seg = torch.nn.CrossEntropyLoss(ignore_index=ignore_label)
        self.weight = weight


    def forward(self, feats):
        # print(len(feats))
        # print(feats[0].shape)
        # print(feats[1].shape)
        # print(feats[2].shape)
        # print(feats[3].shape)
        # output = self.main(feats[0])
        # print(len(feats))
        # print(feats[-1].size())
        output = self.aspp(feats[-1])
        return output

    def init_weights(self):
        def _init_weights(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                m.weight.data.normal_(0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)
        self.aspp.apply(_init_weights)

    def loss(self,
             logits,
             gt_segs,
             weight=1.0):
        losses = dict()
        # print(pred_segs.shape)
        # print(gt_segs.shape)
        # print(torch.squeeze(gt_segs, dim=1))
        if isinstance(logits, list):
            iter_loss = 0
            for logit in logits:
                _, _, H, W = logit.shape
                # size – The requested size in pixels, as a 2-tuple: (width, height).
                labels_ = _resize_labels(gt_segs, size=(W, H))
                # print(logit.shape)
                # print(labels_.shape)
                iter_loss += weight * self.loss_seg(logit, torch.squeeze(labels_.to(logit.device), dim=1))
            losses['loss_seg'] = iter_loss
        else:
            _, _, H, W = logits.shape
            # size – The requested size in pixels, as a 2-tuple: (width, height).
            labels_ = _resize_labels(gt_segs, size=(W, H))
            losses['loss_seg'] = weight * self.loss_seg(logits, torch.squeeze(labels_.to(logits.device), dim=1))
        # print(type(losses))

        # losses['loss_seg'] = weight * self.loss_seg(logits, torch.squeeze(gt_segs, dim=1))

        return losses

def _resize_labels(labels, size):
    """
    Downsample labels for 0.5x and 0.75x logits by nearest interpolation.
    Other nearest methods result in misaligned labels.
    -> F.interpolate(labels, shape, mode='nearest')
    -> cv2.resize(labels, shape, interpolation=cv2.INTER_NEAREST)
    """
    new_labels = []
    for label in labels:
        label = label.detach().cpu().float().numpy()
        label = Image.fromarray(label).resize(size, resample=Image.NEAREST)
        new_labels.append(np.asarray(label))
    new_labels = torch.LongTensor(new_labels)
    return new_labels
 


class _ASPP(nn.Module):
    """
    Atrous spatial pyramid pooling (ASPP)
    """

    def __init__(self, in_ch, out_ch, rates):
        super(_ASPP, self).__init__()
        for i, rate in enumerate(rates):
            self.add_module(
                "c{}".format(i),
                nn.Conv2d(in_ch, out_ch, 3, 1, padding=rate, dilation=rate, bias=True),
            )

        for m in self.children():
            nn.init.normal_(m.weight, mean=0, std=0.01)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return sum([stage(x) for stage in self.children()])


if __name__ == '__main__':
    decoder = SegDecoder()
    feats = torch.from_numpy(np.random.randn(10, 256, 13, 13).astype(np.float32))
    # print(feats)
    logits = decoder(feats)
    print(logits)
    print(logits.shape)
