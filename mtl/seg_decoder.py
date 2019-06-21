import torch
import torch.nn as nn
import numpy as np


class SegDecoder(nn.Module):
    def __init__(self):
        super(SegDecoder, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(256, 2048, 4, 1, 0, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(2048, 1024, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(256, 182, 4, 2, 1, bias=False),
        )

    def forward(self, feats):
        output = self.main(feats)
        return output


if __name__ == '__main__':
    decoder = SegDecoder()
    feats = torch.from_numpy(np.random.randn(10, 256, 13, 13).astype(np.float32))
    # print(feats)
    logits = decoder(feats)
    print(logits)
    print(logits.shape)
