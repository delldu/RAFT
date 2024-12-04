import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import todos
import pdb

class FlowHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv2 = nn.Conv2d(256, 2, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # tensor [x] size: [1, 128, 55, 128], min: -1.0, max: 1.0, mean: -0.007441
        x = self.conv2(self.relu(self.conv1(x)))
        # tensor [x] size: [1, 2, 55, 128], min: -2.050006, max: 8.399508, mean: 0.129837
        return x


class SepConvGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.convz1 = nn.Conv2d(384, 128, (1, 5), padding=(0, 2))
        self.convr1 = nn.Conv2d(384, 128, (1, 5), padding=(0, 2))
        self.convq1 = nn.Conv2d(384, 128, (1, 5), padding=(0, 2))

        self.convz2 = nn.Conv2d(384, 128, (5, 1), padding=(2, 0))
        self.convr2 = nn.Conv2d(384, 128, (5, 1), padding=(2, 0))
        self.convq2 = nn.Conv2d(384, 128, (5, 1), padding=(2, 0))

    def forward(self, h, x):
        # tensor [h] size: [1, 128, 55, 128], min: -1.0, max: 1.0, mean: 0.046708
        # tensor [x] size: [1, 256, 55, 128], min: -1.597288, max: 6.42285, mean: 0.084316

        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q
        # tensor [h] size: [1, 128, 55, 128], min: -1.0, max: 1.0, mean: 0.046477

        return h


class MotionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # corr_levels = 4
        # corr_radius = 4
        # cor_planes = corr_levels * (2 * corr_radius + 1) ** 2
        cor_planes = 324
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64 + 192, 128 - 2, 3, padding=1)

    def forward(self, flow, corr):
        # tensor [flow] size: [1, 2, 55, 128], min: -1.508427, max: 2.091426, mean: 0.32261
        # tensor [corr] size: [1, 324, 55, 128], min: -4.550781, max: 21.899391, mean: 0.168806

        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        motion_feat = torch.cat([out, flow], dim=1)

        # tensor [motion_feat] size: [1, 128, 55, 128], min: -1.508427, max: 6.19675, mean: 0.129374
        return motion_feat


class BasicUpdateBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = MotionEncoder()
        self.gru = SepConvGRU()
        self.flow_head = FlowHead()

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), 
            nn.ReLU(), 
            nn.Conv2d(256, 64 * 9, 1, padding=0)
        )

    def forward(self, net, inp, corr, flow) -> List[torch.Tensor]:
        motion_feat = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_feat], dim=1)

        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        # scale mask to balence gradients
        mask = 0.25 * self.mask(net)
        return net, mask, delta_flow
