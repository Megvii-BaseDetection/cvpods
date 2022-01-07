# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Wrappers around on some nn functions, mainly to support empty tensors.

Ideally, add support directly in PyTorch to empty tensors in those functions.

These can be removed once https://github.com/pytorch/pytorch/issues/12013
is implemented
"""

import math

import torch
from torch.functional import Tensor
from torch.nn import functional as F
from torch.nn.modules.utils import _ntuple

TORCH_VERSION = tuple(int(x) for x in torch.__version__.split(".")[:2])

__all__ = [
    "BatchNorm1d",
    "BatchNorm2d",
    "Conv2d",
    "Conv2dSamePadding",
    "ConvTranspose2d",
    "DisAlignLinear",
    "DisAlignNormalizedLinear",
    "MaxPool2dSamePadding",
    "NormalizedConv2d",
    "NormalizedLinear",
    "SeparableConvBlock",
    "cat",
    "interpolate",
]


def cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


class _NewEmptyTensorOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad):
        shape = ctx.shape
        return _NewEmptyTensorOp.apply(grad, shape), None


class Conv2d(torch.nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        if x.numel() == 0 and self.training:
            # https://github.com/pytorch/pytorch/issues/12013
            assert not isinstance(
                self.norm, torch.nn.SyncBatchNorm
            ), "SyncBatchNorm does not support empty inputs!"

        if x.numel() == 0 and TORCH_VERSION <= (1, 4):
            assert not isinstance(
                self.norm, torch.nn.GroupNorm
            ), "GroupNorm does not support empty inputs in PyTorch <=1.4!"
            # When input is empty, we want to return a empty tensor with "correct" shape,
            # So that the following operations will not panic
            # if they check for the shape of the tensor.
            # This computes the height and width of the output tensor
            output_shape = [
                (i + 2 * p - (di * (k - 1) + 1)) // s + 1
                for i, p, di, k, s in zip(
                    x.shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride
                )
            ]
            output_shape = [x.shape[0], self.weight.shape[0]] + output_shape
            empty = _NewEmptyTensorOp.apply(x, output_shape)
            if self.training:
                # This is to make DDP happy.
                # DDP expects all workers to have gradient w.r.t the same set of parameters.
                _dummy = sum(x.view(-1)[0] for x in self.parameters()) * 0.0
                return empty + _dummy
            else:
                return empty

        x = super().forward(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class Conv2dSamePadding(torch.nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support "SAME" padding mode and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)

        # parse padding mode
        self.padding_method = kwargs.pop("padding", None)
        if self.padding_method is None:
            if len(args) >= 5:
                self.padding_method = args[4]
            else:
                self.padding_method = 0  # default padding number

        if isinstance(self.padding_method, str):
            if self.padding_method.upper() == "SAME":
                # If the padding mode is `SAME`, it will be manually padded
                super().__init__(*args, **kwargs, padding=0)
                # stride
                if isinstance(self.stride, int):
                    self.stride = [self.stride] * 2
                elif len(self.stride) == 1:
                    self.stride = [self.stride[0]] * 2
                # kernel size
                if isinstance(self.kernel_size, int):
                    self.kernel_size = [self.kernel_size] * 2
                elif len(self.kernel_size) == 1:
                    self.kernel_size = [self.kernel_size[0]] * 2
                # dilation
                if isinstance(self.dilation, int):
                    self.dilation = [self.dilation] * 2
                elif len(self.dilation) == 1:
                    self.dilation = [self.dilation[0]] * 2
            else:
                raise ValueError("Unknown padding method: {}".format(self.padding_method))
        else:
            super().__init__(*args, **kwargs, padding=self.padding_method)
        self.norm = norm
        self.activation = activation

    def forward(self, x):
        if isinstance(self.padding_method, str):
            if self.padding_method.upper() == "SAME":
                input_h, input_w = x.shape[-2:]
                stride_h, stride_w = self.stride
                kernel_size_h, kernel_size_w = self.kernel_size
                dilation_h, dilation_w = self.dilation

                output_h = math.ceil(input_h / stride_h)
                output_w = math.ceil(input_w / stride_w)

                padding_needed_h = max(
                    0, (output_h - 1) * stride_h + (kernel_size_h - 1) * dilation_h + 1 - input_h
                )
                padding_needed_w = max(
                    0, (output_w - 1) * stride_w + (kernel_size_w - 1) * dilation_w + 1 - input_w
                )

                left = padding_needed_w // 2
                right = padding_needed_w - left
                top = padding_needed_h // 2
                bottom = padding_needed_h - top

                x = F.pad(x, [left, right, top, bottom])
            else:
                raise ValueError("Unknown padding method: {}".format(self.padding_method))

        x = super().forward(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class MaxPool2dSamePadding(torch.nn.MaxPool2d):
    """
    A wrapper around :class:`torch.nn.MaxPool2d` to support "SAME" padding mode and more features.

    See: https://github.com/pytorch/pytorch/issues/3867
    """

    def __init__(self, *args, **kwargs):
        # parse padding mode
        self.padding_method = kwargs.pop("padding", None)
        if self.padding_method is None:
            if len(args) >= 3:
                self.padding_method = args[2]
            else:
                self.padding_method = 0  # default padding number

        if isinstance(self.padding_method, str):
            if self.padding_method.upper() == "SAME":
                # If the padding mode is `SAME`, it will be manually padded
                super().__init__(*args, **kwargs, padding=0)
                # stride
                if isinstance(self.stride, int):
                    self.stride = [self.stride] * 2
                elif len(self.stride) == 1:
                    self.stride = [self.stride[0]] * 2
                # kernel size
                if isinstance(self.kernel_size, int):
                    self.kernel_size = [self.kernel_size] * 2
                elif len(self.kernel_size) == 1:
                    self.kernel_size = [self.kernel_size[0]] * 2
            else:
                raise ValueError("Unknown padding method: {}".format(self.padding_method))
        else:
            super().__init__(*args, **kwargs, padding=self.padding_method)

    def forward(self, x):
        if isinstance(self.padding_method, str):
            if self.padding_method.upper() == "SAME":
                input_h, input_w = x.shape[-2:]
                stride_h, stride_w = self.stride
                kernel_size_h, kernel_size_w = self.kernel_size

                output_h = math.ceil(input_h / stride_h)
                output_w = math.ceil(input_w / stride_w)

                padding_needed_h = max(
                    0, (output_h - 1) * stride_h + (kernel_size_h - 1) + 1 - input_h
                )
                padding_needed_w = max(
                    0, (output_w - 1) * stride_w + (kernel_size_w - 1) + 1 - input_w
                )

                left = padding_needed_w // 2
                right = padding_needed_w - left
                top = padding_needed_h // 2
                bottom = padding_needed_h - top

                x = F.pad(x, [left, right, top, bottom])
            else:
                raise ValueError("Unknown padding method: {}".format(self.padding_method))

        x = super().forward(x)
        return x


class NormalizedConv2d(torch.nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        # add for support NormalizedConv2d
        feat_norm = kwargs.pop("feat_norm", None)
        scale_mode = kwargs.pop("scale_mode", None)
        scale_init = kwargs.pop("scale_init", None)

        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

        self.scale_mode = scale_mode
        self.scale_init = scale_init
        self.feat_norm = feat_norm

        if self.scale_mode == 'constant':
            self.scale = scale_init
        elif self.scale_mode == 'learn':
            self.scale = torch.nn.Parameter(torch.ones(1) * scale_init)
        else:
            raise NotImplementedError

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'

        s += ', feat_norm={feat_norm}'
        s += ', scale_mode={scale_mode}'
        s += ', scale_init={scale_init}'

        return s.format(**self.__dict__)

    def forward(self, x):
        if x.numel() == 0 and self.training:
            # https://github.com/pytorch/pytorch/issues/12013
            assert not isinstance(
                self.norm, torch.nn.SyncBatchNorm
            ), "SyncBatchNorm does not support empty inputs!"

        if x.numel() == 0 and TORCH_VERSION <= (1, 4):
            assert not isinstance(
                self.norm, torch.nn.GroupNorm
            ), "GroupNorm does not support empty inputs in PyTorch <=1.4!"
            # When input is empty, we want to return a empty tensor with "correct" shape,
            # So that the following operations will not panic
            # if they check for the shape of the tensor.
            # This computes the height and width of the output tensor
            output_shape = [
                (i + 2 * p - (di * (k - 1) + 1)) // s + 1
                for i, p, di, k, s in zip(
                    x.shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride
                )
            ]
            output_shape = [x.shape[0], self.weight.shape[0]] + output_shape
            empty = _NewEmptyTensorOp.apply(x, output_shape)
            if self.training:
                # This is to make DDP happy.
                # DDP expects all workers to have gradient w.r.t the same set of parameters.
                _dummy = sum(x.view(-1)[0] for x in self.parameters()) * 0.0
                return empty + _dummy
            else:
                return empty

        # feature normalization with l2-norm
        if self.feat_norm:
            x = x.div(torch.norm(x, p=2, dim=1, keepdim=True) + 1e-6)
        # weight normalization with l2-norm
        self.weight.data = self.weight.data.div(
            torch.norm(self.weight.data, p=2, dim=1, keepdim=True)
        )

        x = super().forward(x)
        # add scale for prediction
        x = self.scale * x

        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class NormalizedLinear(torch.nn.Module):
    """
    A advanced Linear layer which supports weight normalization or cosine normalization.

    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=False,
        feat_norm=True,
        scale_mode='learn',
        scale_init=1.0
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.feat_norm = feat_norm
        self.scale_mode = scale_mode
        self.scale_init = scale_init

        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        if self.scale_mode == 'constant':
            self.scale = scale_init
        elif self.scale_mode == 'learn':
            self.scale = torch.nn.Parameter(torch.ones(1) * scale_init)
        else:
            raise NotImplementedError

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs):
        """
        Args:
            inputs (torch.Tensor): (N, C)
        Return:
            output (torch.Tensor): (N, D)
        """
        if self.feat_norm:
            inputs = torch.nn.functional.normalize(inputs, dim=1)

        output = inputs.mm(torch.nn.functional.normalize(self.weight, dim=1).t())
        output = self.scale * output
        return output

    def extra_repr(self):
        s = ('in_features={in_features}, out_features={out_features}')
        if self.bias is None:
            s += ', bias=False'
        s += ', feat_norm={feat_norm}'
        s += ', scale_mode={scale_mode}'
        s += ', scale_init={scale_init}'

        return s.format(**self.__dict__)


class DisAlignLinear(torch.nn.Linear):
    """
    A wrapper for nn.Linear with support of DisAlign method.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.confidence_layer = torch.nn.Linear(in_features, 1)
        self.logit_scale = torch.nn.Parameter(torch.ones(1, out_features))
        self.logit_bias = torch.nn.Parameter(torch.zeros(1, out_features))
        torch.nn.init.constant_(self.confidence_layer.weight, 0.1)

    def forward(self, input: Tensor):
        logit_before = F.linear(input, self.weight, self.bias)
        confidence = self.confidence_layer(input).sigmoid()
        logit_after = (1 + confidence * self.logit_scale) * logit_before + \
            confidence * self.logit_bias
        return logit_after


class DisAlignNormalizedLinear(NormalizedLinear):
    """
    A wrapper for nn.Linear with support of DisAlign method.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, **args) -> None:
        super().__init__(in_features=in_features, out_features=out_features, bias=bias, **args)
        self.confidence_layer = torch.nn.Linear(in_features, 1)
        self.logit_scale = torch.nn.Parameter(torch.ones(1, out_features))
        self.logit_bias = torch.nn.Parameter(torch.zeros(1, out_features))
        torch.nn.init.constant_(self.confidence_layer.weight, 0.1)

    def forward(self, input: Tensor):
        if self.feat_norm:
            input = torch.nn.functional.normalize(input, dim=1)

        output = input.mm(torch.nn.functional.normalize(self.weight, dim=1).t())
        logit_before = self.scale * output

        confidence = self.confidence_layer(input).sigmoid()
        logit_after = (1 + confidence * self.logit_scale) * logit_before + \
            confidence * self.logit_bias
        return logit_after


class SeparableConvBlock(torch.nn.Module):
    """
    Depthwise seperable convolution block.
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=True,
                 norm=None, activation=None):
        """
        Args:
            in_channels (int): the number of input tensor channels.
            out_channels (int):the number of output tensor channels.
            kernel_size (int): the kernel size.
            stride (int or tuple or list): the stride.
            bias (bool): if `True`, the pointwise conv applies bias.
            apply_bn (bool): if `True`, apply BN layer after conv layer.
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        super().__init__()
        self.norm = norm
        self.activation = activation
        self.depthwise = Conv2dSamePadding(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = Conv2dSamePadding(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=bias,
        )
        if bias:
            self.bias = self.pointwise.bias

    def forward(self, inputs):
        x = self.depthwise(inputs)
        x = self.pointwise(x)

        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


if TORCH_VERSION > (1, 4):
    ConvTranspose2d = torch.nn.ConvTranspose2d
else:

    class ConvTranspose2d(torch.nn.ConvTranspose2d):
        """
        A wrapper around :class:`torch.nn.ConvTranspose2d` to support zero-size tensor.
        """

        def forward(self, x):
            if x.numel() > 0:
                return super(ConvTranspose2d, self).forward(x)
            # get output shape

            # When input is empty, we want to return a empty tensor with "correct" shape,
            # So that the following operations will not panic
            # if they check for the shape of the tensor.
            # This computes the height and width of the output tensor
            output_shape = [
                (i - 1) * d - 2 * p + (di * (k - 1) + 1) + op
                for i, p, di, k, d, op in zip(
                    x.shape[-2:],
                    self.padding,
                    self.dilation,
                    self.kernel_size,
                    self.stride,
                    self.output_padding,
                )
            ]
            output_shape = [x.shape[0], self.out_channels] + output_shape
            # This is to make DDP happy.
            # DDP expects all workers to have gradient w.r.t the same set of parameters.
            _dummy = sum(x.view(-1)[0] for x in self.parameters()) * 0.0
            return _NewEmptyTensorOp.apply(x, output_shape) + _dummy


if TORCH_VERSION > (1, 4):
    BatchNorm2d = torch.nn.BatchNorm2d
else:

    class BatchNorm2d(torch.nn.BatchNorm2d):
        """
        A wrapper around :class:`torch.nn.BatchNorm2d` to support zero-size tensor.
        """

        def forward(self, x):
            if x.numel() > 0:
                return super(BatchNorm2d, self).forward(x)
            # get output shape
            output_shape = x.shape
            return _NewEmptyTensorOp.apply(x, output_shape)


if TORCH_VERSION > (1, 4):
    BatchNorm1d = torch.nn.BatchNorm1d
else:

    class BatchNorm1d(torch.nn.BatchNorm1d):
        """
        A wrapper around :class:`torch.nn.BatchNorm2d` to support zero-size tensor.
        """

        def forward(self, x):
            if x.numel() > 0:
                return super(BatchNorm1d, self).forward(x)
            # get output shape
            output_shape = x.shape
            return _NewEmptyTensorOp.apply(x, output_shape)


def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    """
    A wrapper around :func:`torch.nn.functional.interpolate` to support zero-size tensor.
    """
    if input.numel() > 0:
        return torch.nn.functional.interpolate(
            input, size, scale_factor, mode, align_corners=align_corners
        )

    def _check_size_scale_factor(dim):
        if size is None and scale_factor is None:
            raise ValueError("either size or scale_factor should be defined")
        if size is not None and scale_factor is not None:
            raise ValueError("only one of size or scale_factor should be defined")
        if (
            scale_factor is not None
            and isinstance(scale_factor, tuple)
            and len(scale_factor) != dim
        ):
            raise ValueError(
                "scale_factor shape must match input shape. "
                "Input is {}D, scale_factor size is {}".format(dim, len(scale_factor))
            )

    def _output_size(dim):
        _check_size_scale_factor(dim)
        if size is not None:
            return size
        scale_factors = _ntuple(dim)(scale_factor)
        # math.floor might return float in py2.7
        return [int(math.floor(input.size(i + 2) * scale_factors[i])) for i in range(dim)]

    output_shape = tuple(_output_size(2))
    output_shape = input.shape[:-2] + output_shape
    return _NewEmptyTensorOp.apply(input, output_shape)
