import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F

def compute_depth_expectation(prob, depth_values):
    depth_values = depth_values.view(*depth_values.shape, 1, 1)
    depth = torch.sum(prob * depth_values, 1)
    return depth

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvBlock, self).__init__()

        if kernel_size == 3:
            self.conv = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_channels, out_channels, 3, padding=0, stride=1),
            )
        elif kernel_size == 1:
            self.conv = nn.Conv2d(int(in_channels), int(out_channels), 1, padding=0, stride=1)

        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class ConvBlock_double(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvBlock_double, self).__init__()

        if kernel_size == 3:
            self.conv = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_channels, out_channels, 3, padding=0, stride=1),
            )
        elif kernel_size == 1:
            self.conv = nn.Conv2d(int(in_channels), int(out_channels), 1, padding=0, stride=1)

        self.nonlin = nn.ELU(inplace=True)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, 1, padding=0, stride=1)
        self.nonlin_2 =nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        out = self.conv_2(out)
        out = self.nonlin_2(out)
        return out
    
class DecoderFeature(nn.Module):
    def __init__(self, feat_channels, num_ch_dec=[64, 64, 128, 256]):
        super(DecoderFeature, self).__init__()
        self.num_ch_dec = num_ch_dec
        self.feat_channels = feat_channels

        self.upconv_3_0 = ConvBlock(self.feat_channels[3], self.num_ch_dec[3], kernel_size=1)
        self.upconv_3_1 = ConvBlock_double(
            self.feat_channels[2] + self.num_ch_dec[3],
            self.num_ch_dec[3],
            kernel_size=1)
        
        self.upconv_2_0 = ConvBlock(self.num_ch_dec[3], self.num_ch_dec[2], kernel_size=3)
        self.upconv_2_1 = ConvBlock_double(
            self.feat_channels[1] + self.num_ch_dec[2],
            self.num_ch_dec[2],
            kernel_size=3)
        
        self.upconv_1_0 = ConvBlock(self.num_ch_dec[2], self.num_ch_dec[1], kernel_size=3)
        self.upconv_1_1 = ConvBlock_double(
            self.feat_channels[0] + self.num_ch_dec[1],
            self.num_ch_dec[1],
            kernel_size=3)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, ref_feature):
        x = ref_feature[3]

        x = self.upconv_3_0(x)
        x = torch.cat((self.upsample(x), ref_feature[2]), 1)
        x = self.upconv_3_1(x)

        x = self.upconv_2_0(x)
        x = torch.cat((self.upsample(x), ref_feature[1]), 1)
        x = self.upconv_2_1(x)

        x = self.upconv_1_0(x)
        x = torch.cat((self.upsample(x), ref_feature[0]), 1)
        x = self.upconv_1_1(x)
        return x


class UNet(nn.Module):
    def __init__(self, inp_ch=32, output_chal=1, down_sample_times=3, channel_mode='v0'):
        super(UNet, self).__init__()
        basic_block = ConvBnReLU
        num_depth = 128

        self.conv0 = basic_block(inp_ch, num_depth)
        if channel_mode == 'v0':
            channels = [num_depth, num_depth//2, num_depth//4, num_depth//8, num_depth // 8]
        elif channel_mode == 'v1':
            channels = [num_depth, num_depth, num_depth, num_depth, num_depth, num_depth]
        self.down_sample_times = down_sample_times
        for i in range(down_sample_times):
            setattr(
                self, 'conv_%d' % i,
                nn.Sequential(
                    basic_block(channels[i], channels[i+1], stride=2),
                    basic_block(channels[i+1], channels[i+1])
                )
            )
        for i in range(down_sample_times-1,-1,-1):
            setattr(self, 'deconv_%d' % i,
                    nn.Sequential(
                        nn.ConvTranspose2d(
                            channels[i+1],
                            channels[i],
                            kernel_size=3,
                            padding=1,
                            output_padding=1,
                            stride=2,
                            bias=False),
                        nn.BatchNorm2d(channels[i]),
                        nn.ReLU(inplace=True)
                    )
                )
            self.prob = nn.Conv2d(num_depth, output_chal, 1, stride=1, padding=0)
    
    def forward(self, x):
        features = {}
        conv0 = self.conv0(x)
        x = conv0
        features[0] = conv0
        for i in range(self.down_sample_times):
            x = getattr(self, 'conv_%d' % i)(x)
            features[i+1] = x
        for i in range(self.down_sample_times-1,-1,-1):
            x = features[i] + getattr(self, 'deconv_%d' % i)(x)
        x = self.prob(x)
        return x

class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=pad,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class HourglassDecoder(nn.Module):
    def __init__(self, cfg):
        super(HourglassDecoder, self).__init__()
        self.inchannels = cfg.model.decode_head.in_channels #  [256, 512, 1024, 2048]
        self.decoder_channels = cfg.model.decode_head.decoder_channel # [64, 64, 128, 256]
        self.min_val = cfg.data_basic.depth_normalize[0]
        self.max_val = cfg.data_basic.depth_normalize[1]

        self.num_ch_dec = self.decoder_channels # [64, 64, 128, 256]
        self.num_depth_regressor_anchor = 512
        self.feat_channels = self.inchannels
        unet_in_channel = self.num_ch_dec[1]
        unet_out_channel = 256

        self.decoder_mono = DecoderFeature(self.feat_channels, self.num_ch_dec)
        self.conv_out_2 = UNet(inp_ch=unet_in_channel,
                               output_chal=unet_out_channel + 1,
                               down_sample_times=3,
                               channel_mode='v0',
                               )

        self.depth_regressor_2 = nn.Sequential(
            nn.Conv2d(unet_out_channel,
                      self.num_depth_regressor_anchor,
                      kernel_size=3,
                      padding=1,
                ),
            nn.BatchNorm2d(self.num_depth_regressor_anchor),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                self.num_depth_regressor_anchor,
                self.num_depth_regressor_anchor,
                kernel_size=1,
            )
        )
        self.residual_channel = 16
        self.conv_up_2 = nn.Sequential(
            nn.Conv2d(1 + 2 + unet_out_channel, self.residual_channel, 3, padding=1),
            nn.BatchNorm2d(self.residual_channel),
            nn.ReLU(),
            nn.Conv2d(self.residual_channel, self.residual_channel, 3, padding=1),
            nn.Upsample(scale_factor=4),
            nn.Conv2d(self.residual_channel, self.residual_channel, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.residual_channel, 1, 1, padding=0),
        )
    
    def get_bins(self, bins_num):
        depth_bins_vec = torch.linspace(math.log(self.min_val), math.log(self.max_val), bins_num, device='cuda')
        depth_bins_vec = torch.exp(depth_bins_vec)
        return depth_bins_vec
    
    def register_depth_expectation_anchor(self, bins_num, B):
        depth_bins_vec = self.get_bins(bins_num)
        depth_bins_vec = depth_bins_vec.unsqueeze(0).repeat(B, 1)
        self.register_buffer('depth_expectation_anchor', depth_bins_vec, persistent=False)

    def upsample(self, x, scale_factor=2):
        return F.interpolate(x, scale_factor=scale_factor, mode='nearest')

    def regress_depth_2(self, feature_map_d):
        prob = self.depth_regressor_2(feature_map_d).softmax(dim=1)
        B = prob.shape[0]
        if "depth_expectation_anchor" not in self._buffers:
            self.register_depth_expectation_anchor(self.num_depth_regressor_anchor, B)
        d = compute_depth_expectation(
            prob,
            self.depth_expectation_anchor[:B, ...]
        ).unsqueeze(1)
        return d

    def create_mesh_grid(self, height, width, batch, device="cuda", set_buffer=True):
        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=device),
                               torch.arange(0, width, dtype=torch.float32, device=device)], indexing='ij')
        meshgrid = torch.stack((x, y))
        meshgrid = meshgrid.unsqueeze(0).repeat(batch, 1, 1, 1)
        return meshgrid

    def forward(self, features_mono, **kwargs):
        '''
        trans_ref2src: list of transformation matrix from the reference view to source view. [B, 4, 4]
        inv_intrinsic_pool: list of inverse intrinsic matrix.
        features_mono: features of reference and source views. [[ref_f1, ref_f2, ref_f3, ref_f4],[src1_f1, src1_f2, src1_f3, src1_f4], ...].
        '''
        outputs = {}
        # get encoder feature of the reference view
        ref_feat = features_mono

        feature_map_mono = self.decoder_mono(ref_feat)
        feature_map_mono_pred = self.conv_out_2(feature_map_mono)
        confidence_map_2 = feature_map_mono_pred[:, -1:, :, :]
        feature_map_d_2 = feature_map_mono_pred[:, :-1, :, :]

        depth_pred_2 = self.regress_depth_2(feature_map_d_2)

        B, _, H, W = depth_pred_2.shape

        meshgrid = self.create_mesh_grid(H, W, B)

        depth_pred_mono = self.upsample(depth_pred_2, scale_factor=4) + 1e-1 * \
            self.conv_up_2(
                torch.cat((depth_pred_2, meshgrid[:B, ...], feature_map_d_2), 1)
            )
        confidence_map_mono = self.upsample(confidence_map_2, scale_factor=4)

        outputs=dict(
            prediction=depth_pred_mono,
            confidence=confidence_map_mono,
            pred_logit=None,
        )
        return outputs