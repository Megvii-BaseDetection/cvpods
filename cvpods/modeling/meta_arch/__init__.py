#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# This file has been modified by Megvii ("Megvii Modifications").
# All Megvii Modifications are Copyright (C) 2019-2021 Megvii Inc. All rights reserved.

# import all the meta_arch, so they will be registered

from .auto_assign import AutoAssign
from .borderdet import BorderDet
from .boxinst import BoxInst
from .centernet import CenterNet
from .condinst import CondInst
from .dynamic4seg import DynamicNet4Seg
from .efficientdet import EfficientDet
from .fcn import FCNHead
from .fcos import FCOS
from .free_anchor import FreeAnchor
from .panoptic_fpn import PanopticFPN
from .pointrend import CoarseMaskHead, PointRendROIHeads, PointRendSemSegHead, StandardPointHead
from .rcnn import GeneralizedRCNN, ProposalNetwork
from .reppoints import RepPoints
from .retinanet import RetinaNet
from .semantic_seg import SemanticSegmentor, SemSegFPNHead
from .ssd import SSD
from .tensormask import TensorMask
from .yolov3 import YOLOv3
