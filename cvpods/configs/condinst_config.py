#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2019-2021 Megvii Inc. All rights reserved.

from .fcos_config import FCOSConfig

_config_dict = dict(
    MODEL=dict(
        MASK_ON=True,
        SHIFT_GENERATOR=dict(
            NUM_SHIFTS=1,
            OFFSET=0.5,
        ),
        CONDINST=dict(
            MASK_OUT_STRIDE=4,
            MAX_PROPOSALS=500,
            TOPK_PROPOSALS_PER_IM=-1,
            MASK_CENTER_SAMPLE=True,
            INFER_MASK_THRESH=0.5,
            MASK_HEAD=dict(
                HEAD_CHANNELS=8,
                NUM_LAYERS=3,
                DISABLE_REL_COORDS=False
            ),
            MASK_BRANCH=dict(
                IN_FEATURES=["p3", "p4", "p5"],
                BRANCH_CHANNELS=128,
                OUT_CHANNELS=8,
                NORM="BN",
                NUM_CONVS=4
            ),
        ),
        FCOS=dict(
            THRESH_WITH_CENTERNESS=True,
            CENTERNESS_ON_REG=True,
            NORM_REG_TARGETS=True,
            IOU_LOSS_TYPE="giou",
            IOU_SMOOTH=True,
            CENTER_SAMPLING_RADIUS=1.5,
        ),
    ),
)


class CondInstConfig(FCOSConfig):
    def __init__(self, d=None, **kwargs):
        super().__init__(d, **kwargs)
        self._register_configuration(_config_dict)


config = CondInstConfig()
