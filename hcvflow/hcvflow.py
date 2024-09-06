import torch
import torch.nn as nn
import torch.nn.functional as F
from .extractor import FeatureNet, ContextNet
from .correlation import Cost1D, LocalCorr
from .update import BasicUpdateBlock
from utils.utils import coords_grid
from .submodule import *


class Model(nn.Module):
    def __init__(self,
                 downsample_factor=8,
                 feature_channels=256,
                 hidden_dim=128,
                 context_dim=128,
                 corr_radius=4,
                 mixed_precision=False,
                 **kwargs,
                 ):
        super(Model, self).__init__()

        self.downsample_factor = downsample_factor
        self.feature_channels = feature_channels
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.corr_radius = corr_radius
        self.mixed_precision = mixed_precision

        # feature network, context network, and update block
        self.fnet = FeatureNet(output_dim=feature_channels, norm_fn='instance',
                                 )

        self.cnet = ContextNet(output_dim=hidden_dim + context_dim, norm_fn='batch',
                                 )

        # 1D attention
        corr_channels = (2 * corr_radius + 1) * 2 * 8 + (2 * corr_radius + 1) ** 2

        self.conv_down = nn.Conv2d(feature_channels, feature_channels, kernel_size=3, stride=2, padding=1)
        self.CostRegNet = CostRegNet(8, 8)
        self.mask_init = nn.Sequential(
                nn.Conv2d(feature_channels, 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, downsample_factor ** 2 * 9, 1, padding=0))

        # Update block
        self.update_block = BasicUpdateBlock(corr_channels=corr_channels,
                                             hidden_dim=hidden_dim,
                                             context_dim=context_dim,
                                             downsample_factor=downsample_factor,
                                             )

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img, downsample=None):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        n, c, h, w = img.shape
        downsample_factor = self.downsample_factor if downsample is None else downsample
        coords0 = coords_grid(n, h // downsample_factor, w // downsample_factor).to(img.device)
        coords1 = coords_grid(n, h // downsample_factor, w // downsample_factor).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def learned_upflow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        n, _, h, w = flow.shape
        mask = mask.view(n, 1, 9, self.downsample_factor, self.downsample_factor, h, w)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(self.downsample_factor * flow, [3, 3], padding=1)
        up_flow = up_flow.view(n, 2, 9, 1, 1, h, w)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(n, 2, self.downsample_factor * h, self.downsample_factor * w)

    def forward(self, image1, image2, iters=12, flow_init=None, test_mode=False,
                ):
        """ Estimate optical flow between pair of frames """

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        # run the feature network
        fmap1, fmap2 = self.fnet([image1, image2])
        fmap1_down = self.conv_down(fmap1)
        fmap2_down = self.conv_down(fmap2)
        cost_h = build_corr_volume_H(fmap1_down, fmap2_down, flow_h=16, k=8)
        cost_v = build_corr_volume_V(fmap1_down, fmap2_down, flow_v=16, k=8)
        cost_h, flow_h = self.CostRegNet(cost_h)
        cost_v, flow_v = self.CostRegNet(cost_v)
        flow_init = torch.cat((flow_h, flow_v), dim=1)
        cost_fn_1d = Cost1D(cost_h, cost_v, max_flow_h=32, max_flow_v=32, radius=self.corr_radius)
        local_corr = LocalCorr(fmap1, fmap2, radius=self.corr_radius, lr=16)

        # run the context network
        cnet = self.cnet(image1) 

        hdim = self.hidden_dim
        cdim = self.context_dim
        net, inp = torch.split(cnet, [hdim, cdim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)  # 1/8 resolution or 1/4

        if flow_init is not None:  # flow_init is 1/8 resolution or 1/4
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()  # stop gradient
            flow = coords1 - coords0
            cost = cost_fn_1d(flow)
            corr = local_corr(flow)

            cost = torch.cat((cost, corr), dim=1)

            net, up_mask, delta_flow = self.update_block(net, inp, cost, flow,
                                                         upsample=not test_mode or itr == iters - 1,
                                                         )

            coords1 = coords1 + delta_flow

            if test_mode:
                # only upsample the last iteration
                if itr == iters - 1:
                    flow_up = self.learned_upflow(coords1 - coords0, up_mask)
                    return coords1 - coords0, flow_up
            else:
                # upsample predictions

                flow_up = self.learned_upflow(coords1 - coords0, up_mask)
                flow_predictions.append(flow_up)
        
        mask_init = .25 * self.mask_init(fmap1)
        flow_init = self.learned_upflow(flow_init, mask_init)

        return flow_init, flow_predictions

def build_model(args):
    return Model(downsample_factor=args.downsample_factor,
                 feature_channels=args.feature_channels,
                 corr_radius=args.corr_radius,
                 hidden_dim=args.hidden_dim,
                 context_dim=args.context_dim,
                 mixed_precision=args.mixed_precision,
                 )
