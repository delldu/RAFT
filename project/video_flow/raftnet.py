import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .update import BasicUpdateBlock

from typing import List

import todos
import pdb

# ggml_debug -- ggml_grid_sample ?
# https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html

def bilinear_sampler(img, coords):
    # img.size() -- [7040, 1, 55, 128]
    # coords.size() -- [7040, 9, 9, 2]
    B, C, H, W = img.size()
    xgrid, ygrid = coords.split([1, 1], dim=3)

    xgrid = 2.0 * xgrid / (W - 1.0) - 1.0
    ygrid = 2.0 * ygrid / (H - 1.0) - 1.0

    # xgrid.size() -- [7040, 9, 9, 1]
    # ygrid.size() -- [7040, 9, 9, 1]
    grid = torch.cat([xgrid, ygrid], dim=3)
    img = F.grid_sample(img, grid, align_corners=True)

    return img # img.size() -- [7040, 1, 9, 9]


def create_corr_pyramid(fmap1, fmap2) -> List[torch.Tensor]:
    # corr_levels: int, corr_radius: int === 4, 4
    B, C, H, W = fmap1.shape  # (1, 256, 55, 128)
    fmap1 = fmap1.view(B, C, H * W)
    fmap2 = fmap2.view(B, C, H * W)

    # C === 256 ==> torch.sqrt(torch.tensor(C).float()) === 16.0
    # corr = torch.matmul(fmap1.transpose(1, 2), fmap2) / torch.sqrt(torch.tensor(C).float())
    corr = torch.matmul(fmap1.transpose(1, 2), fmap2) / 16.0
    # size() -- [1, 7040, 7040]
    corr = corr.reshape(H * W, 1, H, W)  # ==> size() -- [7040, 1, 55, 128]

    corr_pyramid = []
    corr_pyramid.append(corr)
    corr_levels = 4
    for i in range(corr_levels - 1):
        corr = F.avg_pool2d(corr, 2, stride=2)
        corr_pyramid.append(corr)

    # corr_pyramid is list: len = 4
    #     tensor [item] size: [7040, 1, 55, 128], min: -7.369149, max: 25.431339, mean: 0.033188
    #     tensor [item] size: [7040, 1, 27, 64], min: -3.66336, max: 9.582128, mean: 0.032375
    #     tensor [item] size: [7040, 1, 13, 32], min: -2.107447, max: 4.198452, mean: 0.03262
    #     tensor [item] size: [7040, 1, 6, 16], min: -1.357178, max: 2.21133, mean: 0.03297
    return corr_pyramid

def index_corr_volume(coords, corr_pyramid: List[torch.Tensor], mesh_grid_9x9):
    # tensor [coords] size: [1, 2, 55, 128], min: 0.0, max: 127.0, mean: 45.25
    coords = coords.permute(0, 2, 3, 1) # [1, 2, 55, 128] --> [1, 55, 128, 2]
    # tensor [coords] size: [1, 55, 128, 2], min: 0.0, max: 127.0, mean: 45.25

    B, H, W, N = coords.size()

    out_pyramid = []
    corr_levels = 4
    for i in range(corr_levels): # 4 
        centroid = coords.reshape(B*H*W, 1, 1, 2) / 2**i
        corr = bilinear_sampler(corr_pyramid[i], centroid + mesh_grid_9x9).view(B, H, W, -1)
        out_pyramid.append(corr)

    out = torch.cat(out_pyramid, dim=3)  # [1, 55, 128, 324]
    return out.permute(0, 3, 1, 2)  # [1, 324, 55, 128]

# ggml_debug, ggml_grid(int a, n), a->ne[0] > 0
def coords_grid(B: int, H: int, W: int):
    coords = torch.meshgrid(torch.arange(W), torch.arange(H), indexing="xy")
    # coords[0] -- x
    # tensor([[  0,   1,   2,  ..., 125, 126, 127],
    #         [  0,   1,   2,  ..., 125, 126, 127],
    #         [  0,   1,   2,  ..., 125, 126, 127],
    #         ...,
    #         [  0,   1,   2,  ..., 125, 126, 127],
    #         [  0,   1,   2,  ..., 125, 126, 127],
    #         [  0,   1,   2,  ..., 125, 126, 127]])
    # coords[1]
    # tensor([[ 0,  0,  0,  ...,  0,  0,  0],
    #         [ 1,  1,  1,  ...,  1,  1,  1],
    #         [ 2,  2,  2,  ...,  2,  2,  2],
    #         ...,
    #         [52, 52, 52,  ..., 52, 52, 52],
    #         [53, 53, 53,  ..., 53, 53, 53],
    #         [54, 54, 54,  ..., 54, 54, 54]])

    # ggml_debug -- ggml_stack(int n, t1, t2, ..., dim) ...
    coords = torch.stack(coords, dim=0).float()
    return coords[None].repeat(B, 1, 1, 1) # [1, 2, 55, 128]

class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn="batch", stride=1):
        super().__init__()
        # in_planes = 64
        # planes = 96
        # norm_fn = 'instance'
        # stride = 2

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        assert norm_fn == "batch" or norm_fn == "instance"
        assert stride == 1 or stride == 2

        if norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(planes, track_running_stats=True)
            self.norm2 = nn.BatchNorm2d(planes, track_running_stats=True)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes, track_running_stats=True)
        else:  # norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes, track_running_stats=False)
            self.norm2 = nn.InstanceNorm2d(planes, track_running_stats=False)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes, track_running_stats=False)

        if stride == 1:
            self.downsample = nn.Identity()
        else:
            self.downsample = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        x = self.downsample(x)

        return self.relu(x + y)


class BasicEncoder(nn.Module):
    def __init__(self, norm_fn="batch"):
        super().__init__()
        self.norm_fn = norm_fn
        assert norm_fn == "batch" or norm_fn == "instance"

        if self.norm_fn == "batch": # True
            self.norm1 = nn.BatchNorm2d(64, track_running_stats=True)
        elif self.norm_fn == "instance": # True
            self.norm1 = nn.InstanceNorm2d(64, track_running_stats=False)
        else:
            self.norm1 = nn.Identity()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=2)
        self.layer3 = self._make_layer(128, stride=2)

        # output convolution
        self.conv2 = nn.Conv2d(128, 256, kernel_size=1)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        # tensor [x] size: [2, 3, 440, 1024], min: -1.0, max: 1.0, mean: -0.291007
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.conv2(x)

        return x


class RAFT(nn.Module):
    def __init__(self):
        super(RAFT, self).__init__()
        self.MAX_H = 1024
        self.MAX_W = 1024
        self.MAX_TIMES = 8
        # GPU -- 3.6G, 650ms

        self.hidden_dim = 128
        self.context_dim = 128

        self.cnet = BasicEncoder(norm_fn="batch")
        self.fnet = BasicEncoder(norm_fn="instance")

        self.update_block = BasicUpdateBlock()

        corr_radius = 4
        dy = torch.linspace(-corr_radius, corr_radius, 2 * corr_radius + 1)
        dx = torch.linspace(-corr_radius, corr_radius, 2 * corr_radius + 1)
        mesh_grid_9x9 = torch.stack(torch.meshgrid(dy, dx, indexing="ij"), dim=2) # size() -- [9, 9, 2]
        # tensor [self.mesh_grid_9x9] size: [1, 9, 9, 2], min: -4.0, max: 4.0, mean: 0.0
        self.load_weights()
        self.register_buffer('mesh_grid_9x9', mesh_grid_9x9)
        # torch.save(self.state_dict(), "/tmp/a.pth")
        # from ggml_engine import create_network
        # create_network(self)


    def load_weights(self, model_path="models/video_flow.pth"):
        cdir = os.path.dirname(__file__)
        checkpoint = model_path if cdir == "" else cdir + "/" + model_path
        sd = torch.load(checkpoint)
        new_sd = {}
        for k, v in sd.items():
            k = k.replace("module.", "")
            new_sd[k] = v

        self.load_state_dict(new_sd)

    def resize_pad(self, x):
        # Need Resize ?
        B, C, H, W = x.size()
        # s * H <= MAX_H
        # s * W <= MAX_W
        # ==> s <= min(MAX_H/H, MAX_W/W)
        s = min(min(self.MAX_H / H, self.MAX_W / W), 1.0)
        SH, SW = int(s * H), int(s * W)
        resize_x = F.interpolate(x, size=(SH, SW), mode="bilinear", align_corners=False).to(x.dtype)

        # Need Pad ?
        pad_h, pad_w = resize_x.size(2), resize_x.size(3)  # === SH, SW
        # pad_h:  802 pad_w:  1024 for self.MAX_TIMES == 8 ?
        r_pad = (self.MAX_TIMES - (pad_w % self.MAX_TIMES)) % self.MAX_TIMES
        b_pad = (self.MAX_TIMES - (pad_h % self.MAX_TIMES)) % self.MAX_TIMES
        l_pad = r_pad // 2
        r_pad = r_pad - l_pad
        t_pad = b_pad // 2
        b_pad = b_pad - t_pad
        resize_pad_x = F.pad(resize_x, (l_pad, r_pad, t_pad, b_pad), mode="replicate")

        return resize_pad_x

    def initialize_flow(self, img):
        """Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords = coords_grid(N, H // 8, W // 8).to(img.device)
        return coords

    def upsample_flow(self, flow, mask):
        """Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination"""

        # tensor [flow] size: [1, 2, 55, 128], min: -1.231163, max: 1.291382, mean: -0.110575
        # tensor [mask] size: [1, 576, 55, 128], min: -18.799122, max: 9.778274, mean: -0.933968
        N, _, H, W = flow.shape # [1, 2, 55, 128

        # ggml_debug
        # tensor [flow] size: [1, 2, 55, 128], min: -1.231163, max: 1.291382, mean: -0.110575
        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        # tensor [up_flow] size: [1, 18, 7040], min: -9.849304, max: 10.331055, mean: -0.872977

        # ggml_debug
        mask2 = mask.view(1, 9, 64, H*W)
        mask2 = torch.softmax(mask2, dim=1)
        up_flow2 = up_flow.view(2, 9, 1, H*W)
        up_flow2 = torch.sum(mask2 * up_flow2, dim=1)
        # ==> up_flow2 -- [2, 64, H*W] -> [2, 64, H, W] --> pixel_shuffle --> [2, 1, 8*H, 8*W] --> [1, 2, 8*H, 8*W]
        up_flow2 = F.pixel_shuffle(up_flow2.view(2, 64, H, W), 8)
        # up_flow2 = up_flow2.permute(1, 0, 2, 3)
        return up_flow2.reshape(N, 2, 8 * H, 8 * W)

        # mask = mask.view(N, 1, 9, 8, 8, H, W)
        # mask = torch.softmax(mask, dim=2)
        # # mask.size() -- [1, 1, 9, (8, 8, 55, 128)]
        # up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)
        # up_flow = torch.sum(mask * up_flow, dim=2)


        # delta = (up_flow.view(2, 64, H*W) - up_flow2).abs()
        # todos.debug.output_var("delta", delta)

        # # tensor [up_flow] size: [1, 2, 8, 8, 55, 128], min: -9.720092, max: 8.914838, mean: -0.895681
        # up_flow = up_flow.permute(0, 1, 4, 2, 5, 3) # [1, 2, 8, 8, 55, 128] -> [1, 2, 55, 8, 128, 8]
        # # tensor [up_flow] size: [1, 2, 55, 8, 128, 8], min: -9.720092, max: 8.914838, mean: -0.895681
        # return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def forward(self, image1, image2):
        # todos.debug.output_var("image1", image1)
        # todos.debug.output_var("image2", image2)
        # tensor [image1] size: [1, 3, 436, 1024], min: 0.0, max: 1.0, mean: 0.385581
        # tensor [image2] size: [1, 3, 436, 1024], min: 0.0, max: 1.0, mean: 0.389913

        image1 = self.resize_pad(image1)
        image2 = self.resize_pad(image2)

        image1 = 2 * image1 - 1.0
        image2 = 2 * image2 - 1.0

        forward_flow = self.forward_x(image1, image2)
        backward_flow = -forward_flow # self.forward_x(image2, image1)

        # todos.debug.output_var("forward_flow", forward_flow)
        # todos.debug.output_var("backward_flow", backward_flow)
        # tensor [forward_flow] size: [1, 2, 440, 1024], min: -12.402791, max: 42.499916, mean: 0.977267
        # tensor [backward_flow] size: [1, 2, 440, 1024], min: -43.178162, max: 12.715448, mean: -0.805857
        return torch.cat([forward_flow, backward_flow], dim=0)

    def forward_x(self, image1, image2, iters: int = 20):
        # tensor [image1] size: [1, 3, 440, 1024], min: -1.0, max: 1.0, mean: -0.246679

        """Estimate optical flow between pair of frames"""
        # -----------------------------------------------------------------------------
        B = image1.shape[0]
        images = torch.cat([image1, image2], dim=0) # [2, 3, 440, 1024]

        fmaps = self.fnet(images) # fmaps.size() -- [2, 256, 55, 128]
        fmap1, fmap2 = torch.split(fmaps, [B, B], dim=0) # ==> [1, 256, 55, 128]
        # tensor [fmap1] size: [1, 256, 55, 128], min: -4.488491, max: 4.793384, mean: 0.003023
        # tensor [fmap2] size: [1, 256, 55, 128], min: -4.420507, max: 5.076661, mean: 0.003313

        corr_pyramid = create_corr_pyramid(fmap1, fmap2)

        # corr_pyramid is list: len = 4
        #     tensor [item] size: [7040, 1, 55, 128], min: -7.369149, max: 25.431339, mean: 0.033188
        #     tensor [item] size: [7040, 1, 27, 64], min: -3.66336, max: 9.582128, mean: 0.032375
        #     tensor [item] size: [7040, 1, 13, 32], min: -2.107447, max: 4.198452, mean: 0.03262
        #     tensor [item] size: [7040, 1, 6, 16], min: -1.357178, max: 2.21133, mean: 0.03297

        # -----------------------------------------------------------------------------
        # run the context network
        cnet = self.cnet(image1)
        # tensor [cnet] size: [1, 256, 55, 128], min: -17.80987, max: 14.065307, mean: -0.649572
        net, inp = torch.split(cnet, [self.hidden_dim, self.context_dim], dim=1)
        # [self.hidden_dim, self.context_dim] -- [128, 128]
        # net2 = cnet[:, 0:self.hidden_dim, :, :]
        # inp2 = cnet[:, self.hidden_dim:, :, :]
        # todos.debug.output_var("|net - net2|", (net-net2).abs())
        # todos.debug.output_var("|inp - inp2|", (inp-inp2).abs())

        net = torch.tanh(net)
        inp = torch.relu(inp)
        # -----------------------------------------------------------------------------

        coords0 = self.initialize_flow(image1)
        coords1 = coords0.clone()
        # tensor [coords0] size: [1, 2, 55, 128], min: 0.0, max: 127.0, mean: 45.25

        mesh_grid_9x9 = self.mesh_grid_9x9.to(image1.device)

        m = coords1 - coords0
        for itr in range(iters): # 20
            corr = index_corr_volume(coords1, corr_pyramid, mesh_grid_9x9)
            # tensor [corr] size: [1, 324, 55, 128], min: -4.405475, max: 21.874269, mean: 0.16877
            net, up_mask, delta_flow = self.update_block(net, inp, corr, coords1 - coords0) # coords1 - coords0 === flow
            # F(t+1) = F(t) + \Delta(t)
            coords1 += delta_flow

        # upsample predictions
        flow_up = self.upsample_flow(coords1 - coords0, up_mask) # coords1 - coords0 === flow

        return flow_up  # flow upsample result
