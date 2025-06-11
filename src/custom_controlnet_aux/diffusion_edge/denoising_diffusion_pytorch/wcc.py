from typing import Union, Tuple

import torch
from torch import nn as nn
from torch.nn import functional as F

from custom_controlnet_aux.diffusion_edge.denoising_diffusion_pytorch.quantization import weight_quantize_fn, act_quantize_fn
from custom_controlnet_aux.diffusion_edge.denoising_diffusion_pytorch import wavelet


class WCC(nn.Conv1d):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 stride: Union[int, Tuple] = 1,
                 padding: Union[int, Tuple] = 0,
                 dilation: Union[int, Tuple] = 1,
                 groups: int = 1,
                 bias: bool = False,
                 levels: int = 3,
                 compress_rate: float = 0.25,
                 bit_w: int = 8,
                 bit_a: int = 8,
                 wt_type: str = "db1"):
        super(WCC, self).__init__(in_channels, out_channels, 1, stride, padding, dilation, groups, bias)
        self.layer_type = 'WCC'
        self.bit_w = bit_w
        self.bit_a = bit_a

        self.weight_quant = weight_quantize_fn(self.bit_w)
        self.act_quant = act_quantize_fn(self.bit_a, signed=True)

        self.levels = levels
        self.wt_type = wt_type
        self.compress_rate = compress_rate

        dec_filters, rec_filters = wavelet.create_wavelet_filter(wave=self.wt_type,
                                                                 in_size=in_channels,
                                                                 out_size=out_channels)
        self.wt_filters = nn.Parameter(dec_filters, requires_grad=False)
        self.iwt_filters = nn.Parameter(rec_filters, requires_grad=False)
        self.wt = wavelet.get_transform(self.wt_filters, in_channels, levels)
        self.iwt = wavelet.get_inverse_transform(self.iwt_filters, out_channels, levels)

        self.get_pad = lambda n: ((2 ** levels) - n) % (2 ** levels)

    def forward(self, x):
        in_shape = x.shape
        pads = (0, self.get_pad(in_shape[2]), 0, self.get_pad(in_shape[3]))
        x = F.pad(x, pads)  # pad to match 2^(levels)

        weight_q = self.weight_quant(self.weight)  # quantize weights
        x = self.wt(x)  # H
        topk, ids = self.compress(x)  # T
        topk_q = self.act_quant(topk)  # quantize activations
        topk_q = F.conv1d(topk_q, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)  # K_1x1
        x = self.decompress(topk_q, ids, x.shape)  # T^T
        x = self.iwt(x)  # H^T

        x = x[:, :, :in_shape[2], :in_shape[3]]  # remove pads
        return x

    def compress(self, x):
        b, c, h, w = x.shape
        acc = x.norm(dim=1).pow(2)
        acc = acc.view(b, h * w)
        k = int(h * w * self.compress_rate)
        ids = acc.topk(k, dim=1, sorted=False)[1]
        ids.unsqueeze_(dim=1)
        topk = x.reshape((b, c, h * w)).gather(dim=2, index=ids.repeat(1, c, 1))
        return topk, ids

    def decompress(self, topk, ids, shape):
        b, _, h, w = shape
        ids = ids.repeat(1, self.out_channels, 1)
        x = torch.zeros(size=(b, self.out_channels, h * w), requires_grad=True, device=topk.device)
        x = x.scatter(dim=2, index=ids, src=topk)
        x = x.reshape((b, self.out_channels, h, w))
        return x

    def change_wt_params(self, compress_rate, levels, wt_type="db1"):
        self.compress_rate = compress_rate
        self.levels = levels
        dec_filters, rec_filters = wavelet.create_wavelet_filter(wave=self.wt_type,
                                                                 in_size=self.in_channels,
                                                                 out_size=self.out_channels)
        self.wt_filters = nn.Parameter(dec_filters, requires_grad=False)
        self.iwt_filters = nn.Parameter(rec_filters, requires_grad=False)
        self.wt = wavelet.get_transform(self.wt_filters, self.in_channels, levels)
        self.iwt = wavelet.get_inverse_transform(self.iwt_filters, self.out_channels, levels)

    def change_bit(self, bit_w, bit_a):
        self.bit_w = bit_w
        self.bit_a = bit_a
        self.weight_quant.change_bit(bit_w)
        self.act_quant.change_bit(bit_a)

if __name__ == '__main__':
    wcc = WCC(80, 80)
    x = torch.rand(1, 80, 80, 80)
    y = wcc(x)
    pause = 0