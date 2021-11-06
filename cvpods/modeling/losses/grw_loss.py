#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2019-2021 Megvii Inc. All rights reserved.

import torch
from torch.nn import functional as F
from torch.nn.modules.loss import _WeightedLoss
import numpy as np


class GRWCrossEntropyLoss(_WeightedLoss):
    """
    Generalized Reweight Loss, introduced in
    Distribution Alignment: A Unified Framework for Long-tail Visual Recognition
    https://arxiv.org/abs/2103.16370

    """
    __constants__ = ['ignore_index', 'reduction']

    def _init_weights(self, num_samples_list=[], num_classes=1000, exp_scale=1.0):
        assert len(num_samples_list) > 0, "num_samples_list is empty"

        num_shots = np.array(num_samples_list)
        ratio_list = num_shots / np.sum(num_shots)
        exp_reweight = 1 / (ratio_list ** exp_scale)

        exp_reweight = exp_reweight / np.sum(exp_reweight) * num_classes
        exp_reweight = torch.tensor(exp_reweight).float()
        return exp_reweight

    def __init__(
        self,
        size_average=None,
        ignore_index=-100,
        reduce=None,
        reduction='mean',
        num_samples_list=[],
        num_classes=1000,
        exp_scale=1.0,
    ):

        weights_init = self._init_weights(
            num_samples_list=num_samples_list,
            num_classes=num_classes,
            exp_scale=exp_scale)
        super(GRWCrossEntropyLoss, self).__init__(weights_init, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, input, target):
        if self.weight.device != input.device:
            self.weight.to(input.device)

        return F.cross_entropy(
            input,
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction
        )
