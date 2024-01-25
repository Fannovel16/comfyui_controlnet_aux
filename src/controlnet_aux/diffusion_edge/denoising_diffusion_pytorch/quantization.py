import torch
from torch import nn as nn
from torch.nn import Parameter


def weight_quantization(b):
    def uniform_quant(x, b):
        xdiv = x.mul((2 ** b - 1))
        xhard = xdiv.round().div(2 ** b - 1)
        return xhard

    class _pq(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, alpha):
            input.div_(alpha)  # weights are first divided by alpha
            input_c = input.clamp(min=-1, max=1)  # then clipped to [-1,1]
            sign = input_c.sign()
            input_abs = input_c.abs()
            input_q = uniform_quant(input_abs, b).mul(sign)
            ctx.save_for_backward(input, input_q)
            input_q = input_q.mul(alpha)  # rescale to the original range
            return input_q

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()  # grad for weights will not be clipped
            input, input_q = ctx.saved_tensors
            i = (input.abs() > 1.).float()
            sign = input.sign()
            grad_alpha = (grad_output * (sign * i + (input_q - input) * (1 - i))).sum()
            return grad_input, grad_alpha

    return _pq().apply


class weight_quantize_fn(nn.Module):
    def __init__(self, bit_w):
        super(weight_quantize_fn, self).__init__()
        assert bit_w > 0

        self.bit_w = bit_w - 1
        self.weight_q = weight_quantization(b=self.bit_w)
        self.register_parameter('w_alpha', Parameter(torch.tensor(3.0), requires_grad=True))

    def forward(self, weight):
        mean = weight.data.mean()
        std = weight.data.std()
        weight = weight.add(-mean).div(std)  # weights normalization
        weight_q = self.weight_q(weight, self.w_alpha)
        return weight_q

    def change_bit(self, bit_w):
        self.bit_w = bit_w - 1
        self.weight_q = weight_quantization(b=self.bit_w)

def act_quantization(b, signed=False):
    def uniform_quant(x, b=3):
        xdiv = x.mul(2 ** b - 1)
        xhard = xdiv.round().div(2 ** b - 1)
        return xhard

    class _uq(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, alpha):
            input = input.div(alpha)
            input_c = input.clamp(min=-1, max=1) if signed else input.clamp(max=1)
            input_q = uniform_quant(input_c, b)
            ctx.save_for_backward(input, input_q)
            input_q = input_q.mul(alpha)
            return input_q

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()
            input, input_q = ctx.saved_tensors
            i = (input.abs() > 1.).float()
            sign = input.sign()
            grad_alpha = (grad_output * (sign * i + (input_q - input) * (1 - i))).sum()
            grad_input = grad_input * (1 - i)
            return grad_input, grad_alpha

    return _uq().apply

class act_quantize_fn(nn.Module):
    def __init__(self, bit_a, signed=False):
        super(act_quantize_fn, self).__init__()
        self.bit_a = bit_a
        self.signed = signed
        if signed:
            self.bit_a -= 1
        assert bit_a > 0

        self.act_q = act_quantization(b=self.bit_a, signed=signed)
        self.register_parameter('a_alpha', Parameter(torch.tensor(8.0), requires_grad=True))

    def forward(self, x):
        return self.act_q(x, self.a_alpha)

    def change_bit(self, bit_a):
        self.bit_a = bit_a
        if self.signed:
            self.bit_a -= 1
        self.act_q = act_quantization(b=self.bit_a, signed=self.signed)