import torch
import torch.nn as nn
import copy
import torch.nn.functional as F


class Attention1D(nn.Module):
    def __init__(self, in_channels,
                 h_attention=True,
                 r=8
                 ):
        super(Attention1D, self).__init__()

        self.h_attention = h_attention
        self.r = r

        self.query_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, 1)

        # Initialize: https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/transformer.py#L138
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)  # original Transformer initialization

    def forward(self, feature, position=None, value=None):
        b, c, h, w = feature.size()
        feature = feature + position if position is not None else feature
        query = self.query_conv(feature)  # [B, C, H, W]
        key = self.key_conv(feature)  # [B, C, H, W]
        scale_factor = c ** 0.5

        if self.h_attention:
            key = F.pad(key, (self.r, self.r, 0, 0))
            value = F.pad(feature, (self.r, self.r, 0, 0))
            key = key.unfold(dimension=3, size=2*self.r+1, step=1)  # [B, C, H, W, 2*r+1]
            value = value.unfold(dimension=3, size=2*self.r+1, step=1)  # [B, C, H, W, 2*r+1]        
        else:
            key = F.pad(key, (0, 0, self.r, self.r))
            value = F.pad(feature, (0, 0, self.r, self.r))
            key = key.unfold(dimension=2, size=2*self.r+1, step=1)  # [B, C, H, W, 2*r+1]
            value = value.unfold(dimension=2, size=2*self.r+1, step=1)  # [B, C, H, W, 2*r+1]
        scores = torch.sum(query[:, :, :, :, None] * key, dim=1, keepdim=True) / scale_factor # [B, 1, H, W, 9]
        attention = torch.softmax(scores, dim=-1)
        out = torch.sum(attention * value, dim=-1)

        return out
