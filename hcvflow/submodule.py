import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def convbn(in_channels, out_channels, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_channels))


def convbn_3d(in_channels, out_channels, kernel_size, stride, pad):
    return nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=pad, bias=False),
                         nn.BatchNorm3d(out_channels))



class BasicConv(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, bn=True, relu=True, **kwargs):
        super(BasicConv, self).__init__()

        self.relu = relu
        self.use_bn = bn
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.relu:
            x = nn.LeakyReLU()(x)#, inplace=True)
        return x



def correlation(fea1, fea2):
    cost = torch.sum((fea1 * fea2), dim=1, keepdim=True)
    return cost

def build_correlation_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 1, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i])
        else:
            volume[:, :, i, :, :] = correlation(refimg_fea, targetimg_fea)
    volume = volume.contiguous()
    return volume

# def corr_h(fea1, fea2, D, k):
#     corr = torch.sum(fea1.unsqueeze(2) * fea2.unsqueeze(3), dim=1) / torch.sqrt(torch.tensor(D).float())
#     corr_k, ind_k = torch.topk(corr, k, dim=1, largest=True)
#     return corr_k

def corr_h(fea1, fea2, D, k):
    corr = torch.einsum('bchw, bcnw -> bnhw', fea1, fea2) / torch.sqrt(torch.tensor(D).float())
    corr_k, ind_k = torch.topk(corr, k, dim=1, largest=True)
    return corr_k

def build_corr_volume_H(refimg_fea, targetimg_fea, flow_h, k):
    B, D, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, k, 2*flow_h, H, W])
    for i in range(-flow_h, flow_h):
        if i < 0:
            volume[:, :, i+flow_h, :, -i:] = corr_h(refimg_fea[:, :, :, -i:], targetimg_fea[:, :, :, :i], D, k)
        elif i == 0:
            volume[:, :, i+flow_h, :, :] = corr_h(refimg_fea, targetimg_fea, D, k)
        else:
            volume[:, :, i+flow_h, :, :-i] = corr_h(refimg_fea[:, :, :, :-i], targetimg_fea[:, :, :, i:], D, k)
    volume = volume.contiguous()
    return volume

def corr_v(fea1, fea2, D, k):
    corr = torch.einsum('bchw, bchn -> bnhw', fea1, fea2) / torch.sqrt(torch.tensor(D).float())
    corr_k, ind_k = torch.topk(corr, k, dim=1, largest=True)
    return corr_k

def build_corr_volume_V(refimg_fea, targetimg_fea, flow_v, k):
    B, D, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, k, 2*flow_v, H, W])
    for i in range(-flow_v, flow_v):
        if i < 0:
            volume[:, :, i+flow_v, -i:, :] = corr_v(refimg_fea[:, :, -i:, :], targetimg_fea[:, :, :i, :], D, k)
        elif i == 0:
            volume[:, :, i+flow_v, :, :] = corr_v(refimg_fea, targetimg_fea, D, k)
        else:
            volume[:, :, i+flow_v, :-i, :] = corr_v(refimg_fea[:, :, :-i, :], targetimg_fea[:, :, i:, :], D, k)
    volume = volume.contiguous()
    return volume

class FeatureAtt(nn.Module):
    def __init__(self, cv_chan, feat_chan):
        super(FeatureAtt, self).__init__()

        self.feat_att = nn.Sequential(
            BasicConv(feat_chan, feat_chan//2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(feat_chan//2, cv_chan, 1))

    def forward(self, cv, feat):
        feat_att = self.feat_att(feat).unsqueeze(2)
        cv = torch.sigmoid(feat_att)*cv
        return cv

# class CosRegNet(nn.Module):
#     def __init__(self, k, in_channels):
#         super(CosRegNet, self).__init__()

#         self.corr_stem = BasicConv(k, in_channels, is_3d=True, kernel_size=3, stride=1, padding=1)

#         self.conv1 = nn.Sequential(BasicConv(in_channels, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
#                                              padding=1, stride=2, dilation=1),
#                                    BasicConv(in_channels*2, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
#                                              padding=1, stride=1, dilation=1))
                                    
#         self.conv2 = nn.Sequential(BasicConv(in_channels*2, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
#                                              padding=1, stride=2, dilation=1),
#                                    BasicConv(in_channels*4, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
#                                              padding=1, stride=1, dilation=1))                             

#         self.conv2_up = BasicConv(in_channels*4, in_channels*2, deconv=True, is_3d=True, bn=True,
#                                   relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

#         self.conv1_up = BasicConv(in_channels*2, 1, deconv=True, is_3d=True, bn=False,
#                                   relu=False, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

#         # self.agg_0 = nn.Sequential(BasicConv(in_channels*8, in_channels*4, is_3d=True, kernel_size=1, padding=0, stride=1),
#         #                            BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),
#         #                            BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),)

#         self.agg_1 = nn.Sequential(BasicConv(in_channels*4, in_channels*2, is_3d=True, kernel_size=1, padding=0, stride=1),
#                                    BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1),
#                                    BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1))

#         # self.conv = nn.Sequential(BasicConv(in_channels*2, in_channels, is_3d=True, bn=True, relu=True, kernel_size=3,
#         #                                      padding=1, stride=1, dilation=1),
#         #                           nn.Conv3d(in_channels, 1, kernel_size=1))

#         # self.feature_att_8 = FeatureAtt(in_channels, 32)
#         # self.feature_att_16 = FeatureAtt(in_channels*2, 64)
#         # self.feature_att_32 = FeatureAtt(in_channels*4, 96)
#         # self.feature_att_up_16 = FeatureAtt(in_channels*2, 64)

#     def forward(self, x):
#         conv0 = self.corr_stem(x)
#         # conv0 = self.feature_att_8(conv0, features[0])

#         conv1 = self.conv1(conv0)
#         # conv1 = self.feature_att_16(conv1, features[1])

#         conv2 = self.conv2(conv1)
#         # conv2 = self.feature_att_32(conv2, features[2])

#         conv2_up = self.conv2_up(conv2)
#         conv1 = torch.cat((conv2_up, conv1), dim=1)
#         conv1 = self.agg_1(conv1)
#         # conv1 = self.feature_att_up_16(conv1, features[1])

#         cost = self.conv1_up(conv1)
#         # conv0 = torch.cat((conv1_up, conv0), dim=1)
#         # cost = self.conv(conv0)

#         prob = F.softmax(cost.squeeze(1), dim=1)
#         flow_r = prob.shape[1]//2
#         flow_values = torch.arange(-flow_r, flow_r, dtype=prob.dtype, device=prob.device).view(1, 2*flow_r, 1, 1)
#         flow = torch.sum(prob * flow_values, 1, keepdim=True)
#         # torch.cuda.empty_cache()
#         return cost, flow

class CostRegNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CostRegNet, self).__init__()
        mid_channel = 16

        self.conv1 = nn.Sequential(convbn_3d(in_channels, mid_channel * 2, 3, 2, 1),
                                   nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(convbn_3d(mid_channel * 2, mid_channel * 2, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        # self.conv3 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 4, 3, 2, 1),
        #                            nn.ReLU(inplace=True))

        # self.conv4 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 4, 3, 1, 1),
        #                            nn.ReLU(inplace=True))

        # self.conv5 = nn.Sequential(
        #     nn.ConvTranspose3d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
        #     nn.BatchNorm3d(in_channels * 2))

        self.conv3 = nn.Sequential(
            nn.ConvTranspose3d(mid_channel * 2, mid_channel, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(mid_channel))

        self.redir1 = convbn_3d(in_channels, mid_channel, kernel_size=1, stride=1, pad=0)
        # self.redir2 = convbn_3d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, pad=0)

        # self.conv = nn.Conv3d(in_channels, 1, kernel_size=3, padding=1, stride=1, bias=False)

        self.conv_up = BasicConv(mid_channel, out_channels, deconv=True, is_3d=True, bn=False,
                                  relu=False, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv_prob = nn.Conv3d(out_channels, 1, kernel_size=3, padding=1, stride=1, bias=False)
                    


    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        cost = F.relu(self.conv3(conv2) + self.redir1(x), inplace=True)
        cost = self.conv_up(cost)
        prob = self.conv_prob(cost)
        flow_init = flow_regression(prob)
        return cost, flow_init

def flow_regression(x):
    prob = F.softmax(x.squeeze(1), dim=1)
    flow_r = prob.shape[1]//2
    flow_values = torch.arange(-flow_r, flow_r, dtype=prob.dtype, device=prob.device).view(1, 2*flow_r, 1, 1)
    flow = torch.sum(prob * flow_values, 1, keepdim=True)
    return flow