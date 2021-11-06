#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# This file has been modified by Megvii ("Megvii Modifications").
# All Megvii Modifications are Copyright (C) 2019-2021 Megvii Inc. All rights reserved.

# TODO wangfeng02: refine __init__.py
from .backbone import Backbone
from .bifpn import BiFPN, build_efficientnet_bifpn_backbone
from .darknet import Darknet, build_darknet_backbone
from .dynamic_arch import DynamicNetwork, build_dynamic_backbone
from .efficientnet import EfficientNet, build_efficientnet_backbone
from .fpn import FPN
from .mobilenet import InvertedResBlock, MobileNetV2, MobileStem, build_mobilenetv2_backbone
from .resnet import ResNet, ResNetBlockBase, build_resnet_backbone, make_stage
# TODO can expose more resnet blocks after careful consideration
from .shufflenet import ShuffleNetV2, ShuffleV2Block, build_shufflenetv2_backbone
from .snet import SNet, build_snet_backbone
# from .timm_backbone import build_timm_backbone
from .transformer import Transformer
