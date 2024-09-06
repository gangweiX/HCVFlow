import torch
import torch.nn.functional as F
from utils.utils import bilinear_sampler, bilinear_sampler_1d

class LocalCorr:
    def __init__(self, fmap1, fmap2, radius=4, lr=16):
        self.radius = radius
        self.lr = lr
        corr = LocalCorr.corr(fmap1, fmap2, lr)
        b, h, w, _, _ = corr.shape
        self.corr = corr.reshape(b*h*w, 1, 2*lr+1, 2*lr+1)

    def __call__(self, flow):
        r = self.radius
        coords = flow + self.lr
        coords = coords.permute(0, 2, 3, 1)
        b, h, w, _ = coords.shape
        dx = torch.linspace(-r, r, 2*r+1, device=coords.device)
        dy = torch.linspace(-r, r, 2*r+1, device=coords.device)
        delta = torch.stack(torch.meshgrid(dy, dx), axis=-1)
        centroid_lvl = coords.reshape(b*h*w, 1, 1, 2)
        delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)
        coords_lvl = centroid_lvl + delta_lvl
        corr = bilinear_sampler(self.corr, coords_lvl)
        corr = corr.view(b, h, w, -1)
        return corr.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2, lr):
        b, c, h, w = fmap1.shape
        fmap2 = F.pad(fmap2, (lr, lr, 0, 0))
        fmap2 = fmap2.unfold(dimension=3, size=2*lr+1, step=1)  # [B, C, H, W, 2*r+1]
        corr = fmap1.new_zeros([b, h, w, 2*lr+1, 2*lr+1])
        fmap1 = fmap1.unsqueeze(4)
        for i in range(-lr, lr+1):
            if i < 0:

                corr[:, -i:, :, i+lr,:] = (fmap1[:, :, -i:, :, :] * fmap2[:, :, :i, :, :]).sum(dim=1)
            elif i == 0:
                corr[:, :, :, i+lr,:] = (fmap1 * fmap2).sum(dim=1)
            else:

                corr[:, :-i, :, i+lr,:] = (fmap1[:, :, :-i, :, :] * fmap2[:, :, i:, :, :]).sum(dim=1)
        return corr / torch.sqrt(torch.tensor(c).float())



class Corr1D:
    def __init__(self, fmap1, fmap2, radius=4):
        self.radius = radius
        self.fmap1 = fmap1
        self.fmap2 = fmap2
        b, c, h, w = fmap1.shape
        self.c = c


    def __call__(self, coords, horizontal):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        b, h, w, _ = coords.shape
        delta = torch.linspace(-r, r, 2*r+1)
        if horizontal:
            delta_xy = torch.stack((delta, torch.zeros_like(delta)), dim=-1).to(coords.device)  # [2r+1, 2]
            delta_xy = delta_xy.view(1, 2*r+1, 1, 1, 2)
        else:
            delta_xy = torch.stack((torch.zeros_like(delta), delta), dim=-1).to(coords.device)  # [2r+1, 2]
            delta_xy = delta_xy.view(1, 2*r+1, 1, 1, 2)

        coords_lvl = coords.view(b, 1, h, w, 2) + delta_xy
        warped_fmap2 = bilinear_sampler(self.fmap2, coords_lvl.reshape(b, -1, w, 2)) # [b, c, (2*r+1)*2*h, w]
        warped_fmap2 = warped_fmap2.view(b, -1, (2*r+1), h, w)
        corr = (self.fmap1[:, :, None, :, :] * warped_fmap2).sum(dim=1)
        return corr / (self.c ** 0.5)

class Cost1D:
    def __init__(self, cost_h, cost_v, max_flow_h, max_flow_v, radius=4):
        self.num_levels = 1
        self.radius = radius
        self.max_flow_h = max_flow_h
        self.max_flow_v = max_flow_v
        b, c, f_h, h, w = cost_h.shape
        _, _, f_v, _, _ = cost_v.shape

        self.cost_h = cost_h.permute(0, 3, 4, 1, 2).reshape(b*h*w, c, 1, f_h)
        self.cost_v = cost_v.permute(0, 3, 4, 1, 2).reshape(b*h*w, c, 1, f_v)

    def __call__(self, flow):
        r = self.radius
        flow_h = flow[:,0] + self.max_flow_h
        flow_v = flow[:,1] + self.max_flow_v

        batch, _, h1, w1 = flow.shape
        dxy = torch.linspace(-r, r, 2*r+1)
        dxy = dxy.view(1, 1, 2*r+1, 1).to(flow.device)


        x0 = dxy + flow_h.reshape(batch*h1*w1, 1, 1, 1)
        y0 = torch.zeros_like(x0)
        flow_h_lvl = torch.cat([x0,y0], dim=-1)
        cost_h = bilinear_sampler_1d(self.cost_h, flow_h_lvl)
        cost_h = cost_h.view(batch, h1, w1, -1)

        x0 = dxy + flow_v.reshape(batch*h1*w1, 1, 1, 1)
        y0 = torch.zeros_like(x0)
        flow_v_lvl = torch.cat([x0,y0], dim=-1)
        cost_v = bilinear_sampler_1d(self.cost_v, flow_v_lvl)
        cost_v = cost_v.view(batch, h1, w1, -1)

        cost_hv = torch.cat((cost_h, cost_v), dim=-1)
        return cost_hv.permute(0, 3, 1, 2).contiguous().float()
