#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# This file has been modified by Megvii ("Megvii Modifications").
# All Megvii Modifications are Copyright (C) 2019-2021 Megvii Inc. All rights reserved.

from .activation_funcs import MemoryEfficientSwish, Swish
from .batch_norm import (
    FrozenBatchNorm2d,
    NaiveSyncBatchNorm,
    NaiveSyncBatchNorm1d,
    get_activation,
    get_norm
)
from .border_align import BorderAlign, BorderAlignFunc
from .deform_conv import DeformConv, ModulatedDeformConv
from .deform_conv_with_off import DeformConvWithOff, ModulatedDeformConvWithOff
from .mask_ops import paste_masks_in_image
from .nms import *
from .position_encoding import (
    PositionEmbeddingLearned,
    PositionEmbeddingSine,
    position_encoding_dict
)
from .roi_align import ROIAlign, roi_align
from .roi_align_rotated import ROIAlignRotated, roi_align_rotated
from .shape_spec import ShapeSpec
from .swap_align2nat import SwapAlign2Nat, swap_align2nat
from .tree_filter_v2 import TreeFilterV2
from .wrappers import *

## SplAtConv2d must be imported after Conv2d
from .splat import SplAtConv2d  # isort:skip

__all__ = [k for k in globals().keys() if not k.startswith("_")]
