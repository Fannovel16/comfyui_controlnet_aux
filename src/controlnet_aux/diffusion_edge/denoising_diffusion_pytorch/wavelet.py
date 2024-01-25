import pywt
import pywt.data
import torch
from torch import nn
from torch.autograd import Function
import torch.nn.functional as F


def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters


def wt(x, filters, in_size, level):
    _, _, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    res = F.conv2d(x, filters, stride=2, groups=in_size, padding=pad)
    if level > 1:
        res[:, ::4] = wt(res[:, ::4], filters, in_size, level - 1)
    res = res.reshape(-1, 2, h // 2, w // 2).transpose(1, 2).reshape(-1, in_size, h, w)
    return res


def iwt(x, inv_filters, in_size, level):
    _, _, h, w = x.shape
    pad = (inv_filters.shape[2] // 2 - 1, inv_filters.shape[3] // 2 - 1)
    res = x.reshape(-1, h // 2, 2, w // 2).transpose(1, 2).reshape(-1, 4 * in_size, h // 2, w // 2)
    if level > 1:
        res[:, ::4] = iwt(res[:, ::4], inv_filters, in_size, level - 1)
    res = F.conv_transpose2d(res, inv_filters, stride=2, groups=in_size, padding=pad)
    return res


def get_inverse_transform(weights, in_size, level):
    class InverseWaveletTransform(Function):

        @staticmethod
        def forward(ctx, input):
            with torch.no_grad():
                x = iwt(input, weights, in_size, level)
            return x

        @staticmethod
        def backward(ctx, grad_output):
            grad = wt(grad_output, weights, in_size, level)
            return grad, None

    return InverseWaveletTransform().apply


def get_transform(weights, in_size, level):
    class WaveletTransform(Function):

        @staticmethod
        def forward(ctx, input):
            with torch.no_grad():
                x = wt(input, weights, in_size, level)
            return x

        @staticmethod
        def backward(ctx, grad_output):
            grad = iwt(grad_output, weights, in_size, level)
            return grad, None

    return WaveletTransform().apply