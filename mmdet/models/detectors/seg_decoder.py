import torch
import torch.nn as nn
import numpy as np

from mmdet.models.builder import build_loss


class SegDecoder(nn.Module):
    def __init__(self, ignore_label):
        super(SegDecoder, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 4, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(256, 256, 4, 2, 2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            # nn.ConvTranspose2d(256, 256, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(256),
            # nn.ReLU(True),
            # # state size. (ngf*2) x 16 x 16
            # nn.ConvTranspose2d(256, 256, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(256),
            # nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(256, 182, 4, 2, 1, bias=False),
        )
        self.loss_seg = torch.nn.CrossEntropyLoss(ignore_index=ignore_label)

    def forward(self, feats):
        # print(len(feats))
        # print(feats[0].shape)
        # print(feats[1].shape)
        # print(feats[2].shape)
        # print(feats[3].shape)
        output = self.main(feats[0])
        return output

    def init_weights(self):
        def _init_weights(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                m.weight.data.normal_(0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)
        self.main.apply(_init_weights)

    def loss(self,
             pred_segs,
             gt_segs,
             weight=1.0):
        losses = dict()
        # print(pred_segs.shape)
        # print(gt_segs.shape)
        # print(torch.squeeze(gt_segs, dim=1))
        losses['loss_seg'] = 0.001 * weight * self.loss_seg(pred_segs, torch.squeeze(gt_segs, dim=1))
        # print(type(losses))
        return losses


if __name__ == '__main__':
    decoder = SegDecoder()
    feats = torch.from_numpy(np.random.randn(10, 256, 13, 13).astype(np.float32))
    # print(feats)
    logits = decoder(feats)
    print(logits)
    print(logits.shape)
