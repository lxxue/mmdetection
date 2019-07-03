import torch
import torch.nn as nn
import numpy as np

from mmdet.models.builder import build_loss


class DeeplabDecoder(nn.Module):
    def __init__(self, ignore_label, inplanes, num_classes, atrous_rates):
        super(DeeplabDecoder, self).__init__()
        self.aspp = _ASPP(inplanes, num_classes, atrous_rates)
        self.loss_seg = torch.nn.CrossEntropyLoss(ignore_index=ignore_label)


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
             pred_segs,
             gt_segs,
             weight=1.0):
        losses = dict()
        # print(pred_segs.shape)
        # print(gt_segs.shape)
        # print(torch.squeeze(gt_segs, dim=1))
        losses['loss_seg'] = weight * self.loss_seg(pred_segs, torch.squeeze(gt_segs, dim=1))
        # print(type(losses))
        return losses


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
