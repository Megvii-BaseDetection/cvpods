#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2019-2021 Megvii Inc. All rights reserved.

from .condinst_config import CondInstConfig

_config_dict = dict(
    MODEL=dict(
        BOXINST=dict(
            BOTTOM_PIXELS_REMOVED=10,
            PAIRWISE_LOSS=dict(
                ENABLE=True,
                PATCH_SIZE=3,
                DILATION=2,
                THETA=2,
                EDGE_COUNT_ONCE=True,
                COLOR_SIM_THRESH=0.3,
                WARMUP_ITERS=10000,
            ),
        ),
        CONDINST=dict(
            MAX_PROPOSALS=-1,
            TOPK_PROPOSALS_PER_IM=64,
            MASK_BRANCH=dict(
                OUT_CHANNELS=16,
            ),
        ),
    ),
)


class BoxInstConfig(CondInstConfig):
    def __init__(self, d=None, **kwargs):
        super().__init__(d, **kwargs)
        self._register_configuration(_config_dict)


config = BoxInstConfig()
