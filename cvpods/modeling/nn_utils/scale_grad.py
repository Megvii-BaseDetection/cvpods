#!/usr/bin/python3
# -*- coding:utf-8 -*-
from torch.autograd.function import Function


class _ScaleGradient(Function):

    @staticmethod
    def forward(ctx, input, scale):
        ctx.scale = scale
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None
