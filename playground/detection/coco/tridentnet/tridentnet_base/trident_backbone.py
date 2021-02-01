# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch

import cvpods.modeling.nn_utils.weight_init as weight_init
from cvpods.layers import Conv2d, FrozenBatchNorm2d, get_activation, get_norm
from cvpods.modeling import ResNet, ResNetBlockBase, make_stage
from cvpods.modeling.backbone.resnet import (
    AVDBottleneckBlock,
    BasicBlock,
    BasicStem,
    BottleneckBlock,
    DeformBottleneckBlock
)

from .trident_conv import TridentConv

__all__ = ["TridentBottleneckBlock", "make_trident_stage", "build_trident_resnet_backbone"]


class TridentBottleneckBlock(ResNetBlockBase):
    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        bottleneck_channels,
        stride=1,
        num_groups=1,
        norm="BN",
        activation=None,
        stride_in_1x1=False,
        num_branch=3,
        dilations=(1, 2, 3),
        concat_output=False,
        test_branch_idx=-1,
    ):
        """
        Args:
            num_branch (int): the number of branches in TridentNet.
            dilations (tuple): the dilations of multiple branches in TridentNet.
            concat_output (bool): if concatenate outputs of multiple branches in TridentNet.
                Use 'True' for the last trident block.
        """
        super().__init__(in_channels, out_channels, stride)

        assert num_branch == len(dilations)

        self.num_branch = num_branch
        self.concat_output = concat_output
        self.test_branch_idx = test_branch_idx

        if in_channels != out_channels:
            self.shortcut = Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                norm=get_norm(norm, out_channels),
            )
        else:
            self.shortcut = None

        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        self.activation = get_activation(activation)

        self.conv1 = Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=stride_1x1,
            bias=False,
            norm=get_norm(norm, bottleneck_channels),
        )

        self.conv2 = TridentConv(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride_3x3,
            paddings=dilations,
            bias=False,
            groups=num_groups,
            dilations=dilations,
            num_branch=num_branch,
            test_branch_idx=test_branch_idx,
            norm=get_norm(norm, bottleneck_channels),
        )

        self.conv3 = Conv2d(
            bottleneck_channels,
            out_channels,
            kernel_size=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        for layer in [self.conv1, self.conv2, self.conv3, self.shortcut]:
            if layer is not None:  # shortcut can be None
                weight_init.c2_msra_fill(layer)

    def forward(self, x):
        num_branch = self.num_branch if self.training or self.test_branch_idx == -1 else 1
        if not isinstance(x, list):
            x = [x] * num_branch
        out = [self.conv1(b) for b in x]
        out = [self.activation(b) for b in out]

        out = self.conv2(out)
        out = [self.activation(b) for b in out]

        out = [self.conv3(b) for b in out]

        if self.shortcut is not None:
            shortcut = [self.shortcut(b) for b in x]
        else:
            shortcut = x

        out = [out_b + shortcut_b for out_b, shortcut_b in zip(out, shortcut)]
        out = [self.activation(b) for b in out]
        if self.concat_output:
            out = torch.cat(out)
        return out


def make_trident_stage(block_class, num_blocks, first_stride, **kwargs):
    """
    Create a resnet stage by creating many blocks for TridentNet.
    """
    blocks = []
    for i in range(num_blocks - 1):
        blocks.append(block_class(stride=first_stride if i == 0 else 1, **kwargs))
        kwargs["in_channels"] = kwargs["out_channels"]
    blocks.append(block_class(stride=1, concat_output=True, **kwargs))
    return blocks


def build_trident_resnet_backbone(cfg, input_shape):
    """
    Create a ResNet instance from config for TridentNet.

    Returns:
        ResNet: a :class:`ResNet` instance.
    """
    # need registration of new blocks/stems?
    norm = cfg.MODEL.RESNETS.NORM
    activation = cfg.MODEL.RESNETS.ACTIVATION
    deep_stem = cfg.MODEL.RESNETS.DEEP_STEM

    if not deep_stem:
        assert getattr(cfg.MODEL.RESNETS, "RADIX", 1) <= 1, \
            "cfg.MODEL.RESNETS.RADIX > 1: {}".format(cfg.MODEL.RESNETS.RADIX)
    stem = BasicStem(
        in_channels=input_shape.channels,
        out_channels=cfg.MODEL.RESNETS.STEM_OUT_CHANNELS,
        norm=norm,
        activation=activation,
        deep_stem=deep_stem,
    )
    freeze_at = cfg.MODEL.BACKBONE.FREEZE_AT

    if freeze_at >= 1:
        for p in stem.parameters():
            p.requires_grad = False
        stem = FrozenBatchNorm2d.convert_frozen_batchnorm(stem)

    # fmt: off
    out_features         = cfg.MODEL.RESNETS.OUT_FEATURES
    depth                = cfg.MODEL.RESNETS.DEPTH
    num_groups           = cfg.MODEL.RESNETS.NUM_GROUPS
    width_per_group      = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
    bottleneck_channels  = num_groups * width_per_group
    in_channels          = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
    out_channels         = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    stride_in_1x1        = cfg.MODEL.RESNETS.STRIDE_IN_1X1
    res5_dilation        = cfg.MODEL.RESNETS.RES5_DILATION
    num_branch           = cfg.MODEL.TRIDENT.NUM_BRANCH
    branch_dilations     = cfg.MODEL.TRIDENT.BRANCH_DILATIONS
    trident_stage        = cfg.MODEL.TRIDENT.TRIDENT_STAGE
    test_branch_idx      = cfg.MODEL.TRIDENT.TEST_BRANCH_IDX
    # fmt: on
    assert res5_dilation in {1, 2}, "res5_dilation cannot be {}.".format(res5_dilation)

    num_blocks_per_stage = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
    }[depth]

    res_stage_idx = {"res2": 2, "res3": 3, "res4": 4, "res5": 5, "linear": 5}
    out_stage_idx = [res_stage_idx[f] for f in out_features]
    trident_stage_idx = res_stage_idx[trident_stage]
    max_stage_idx = max(out_stage_idx)
    deform_on_per_stage = getattr(cfg.MODEL.RESNETS,
                                  "DEFORM_ON_PER_STAGE",
                                  [False] * (max_stage_idx - 1))

    stages = []

    if depth in [18, 34]:
        assert out_channels == 64, "Must set MODEL.RESNETS.RES2_OUT_CHANNELS = 64 for R18/R34"
        assert not any(
            deform_on_per_stage
        ), "MODEL.RESNETS.DEFORM_ON_PER_STAGE unsupported for R18/R34"
        assert res5_dilation == 1, "Must set MODEL.RESNETS.RES5_DILATION = 1 for R18/R34"
        assert num_groups == 1, "Must set MODEL.RESNETS.NUM_GROUPS = 1 for R18/R34"

    for idx, stage_idx in enumerate(range(2, max_stage_idx + 1)):
        dilation = res5_dilation if stage_idx == 5 else 1
        first_stride = 1 if idx == 0 or (stage_idx == 5 and dilation == 2) else 2
        stage_kargs = {
            "num_blocks": num_blocks_per_stage[idx],
            "first_stride": first_stride,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "norm": norm,
            "activation": activation,
        }
        # Use BasicBlock for R18 and R34.
        if depth in [18, 34]:
            stage_kargs["block_class"] = BasicBlock
        else:
            stage_kargs["bottleneck_channels"] = bottleneck_channels
            stage_kargs["stride_in_1x1"] = stride_in_1x1
            stage_kargs["dilation"] = dilation
            stage_kargs["num_groups"] = num_groups
            if stage_idx == trident_stage_idx:
                assert not deform_on_per_stage[
                    idx
                ], "Not support deformable conv in Trident blocks yet."
                stage_kargs["block_class"] = TridentBottleneckBlock
                stage_kargs["num_branch"] = num_branch
                stage_kargs["dilations"] = branch_dilations
                stage_kargs["test_branch_idx"] = test_branch_idx
                stage_kargs.pop("dilation")
            elif deform_on_per_stage[idx]:
                stage_kargs["block_class"] = DeformBottleneckBlock
                # Use True to use modulated deform_conv (DeformableV2);
                # Use False for DeformableV1.
                stage_kargs["deform_modulated"] = cfg.MODEL.RESNETS.DEFORM_MODULATED
                # Number of groups in deformable conv.
                stage_kargs["deform_num_groups"] = cfg.MODEL.RESNETS.DEFORM_NUM_GROUPS
            elif hasattr(cfg.MODEL.RESNETS, "RADIX"):
                # Radix in ResNeSt
                radix = cfg.MODEL.RESNETS.RADIX
                # Apply avg after conv2 in the BottleBlock
                # When AVD=True, the STRIDE_IN_1X1 should be False
                avd = cfg.MODEL.RESNETS.AVD or (radix > 1)
                # Apply avg_down to the downsampling layer for residual path
                avg_down = cfg.MODEL.RESNETS.AVG_DOWN or (radix > 1)
                # Bottleneck_width in ResNeSt
                bottleneck_width = cfg.MODEL.RESNETS.BOTTLENECK_WIDTH

                stage_kargs["block_class"] = AVDBottleneckBlock
                stage_kargs["avd"] = avd
                stage_kargs["avg_down"] = avg_down
                stage_kargs["radix"] = radix
                stage_kargs["bottleneck_width"] = bottleneck_width
            else:
                stage_kargs["block_class"] = BottleneckBlock
        blocks = (
            make_trident_stage(**stage_kargs)
            if stage_idx == trident_stage_idx
            else make_stage(**stage_kargs)
        )
        in_channels = out_channels
        out_channels *= 2
        bottleneck_channels *= 2

        if freeze_at >= stage_idx:
            for block in blocks:
                block.freeze()
        stages.append(blocks)
    return ResNet(stem, stages, out_features=out_features)
