#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2019-2021 Megvii Inc. All rights reserved.

import numpy as np

import torch
from torch import nn

from cvpods.layers import cat, NormalizedLinear


class DisAlignFastRCNNOutputLayers(nn.Module):
    """
    Two linear layers for predicting Fast R-CNN outputs:
      (1) proposal-to-detection box regression deltas
      (2) classification scores
    """

    def __init__(self, input_size, num_classes, cls_agnostic_bbox_reg, box_dim=4):
        """
        Args:
            input_size (int): channels, or (channels, height, width)
            num_classes (int): number of foreground classes
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            box_dim (int): the dimension of bounding boxes.
                Example box dimensions: 4 for regular XYXY boxes and 5 for rotated XYWHA boxes
        """
        super(DisAlignFastRCNNOutputLayers, self).__init__()

        if not isinstance(input_size, int):
            input_size = np.prod(input_size)

        # The prediction layer for num_classes foreground classes and one background class
        # (hence + 1)
        self.cls_score = nn.Linear(input_size, num_classes + 1)
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        self.bbox_pred = nn.Linear(input_size, num_bbox_reg_classes * box_dim)

        # Magnitude and Margin of DisAlign
        self.logit_scale = nn.Parameter(torch.ones(num_classes))
        self.logit_bias = nn.Parameter(torch.zeros(num_classes))
        # Confidence function
        self.confidence_layer = nn.Linear(input_size, 1)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.confidence_layer.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for layer in [self.cls_score, self.confidence_layer, self.bbox_pred]:
            nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        scores = self.cls_score(x)

        # only adjust the foreground classification scores
        confidence = self.confidence_layer(x).sigmoid()
        scores_tmp = confidence * (scores[:, :-1] * self.logit_scale + self.logit_bias)
        scores_tmp = scores_tmp + (1 - confidence) * scores[:, :-1]

        aligned_scores = cat([scores_tmp, scores[:, -1].view(-1, 1)], dim=1)
        proposal_deltas = self.bbox_pred(x)
        return aligned_scores, proposal_deltas


class CosineFastRCNNOutputLayers(nn.Module):
    """
    Two linear layers for predicting Fast R-CNN outputs:
      (1) proposal-to-detection box regression deltas
      (2) classification scores
    """

    def __init__(
        self,
        input_size,
        num_classes,
        cls_agnostic_bbox_reg,
        box_dim=4,
        scale_mode='learn',
        scale_init=20.0
    ):
        """
        Args:
            input_size (int): channels, or (channels, height, width)
            num_classes (int): number of foreground classes
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            box_dim (int): the dimension of bounding boxes.
                Example box dimensions: 4 for regular XYXY boxes and 5 for rotated XYWHA boxes
        """
        super(CosineFastRCNNOutputLayers, self).__init__()

        if not isinstance(input_size, int):
            input_size = np.prod(input_size)

        # The prediction layer for num_classes foreground classes and one background class
        # (hence + 1)
        self.cls_score = NormalizedLinear(
            input_size,
            num_classes + 1,
            scale_mode=scale_mode,
            scale_init=scale_init)
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        self.bbox_pred = nn.Linear(input_size, num_bbox_reg_classes * box_dim)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for layer in [self.cls_score, self.bbox_pred]:
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        scores = self.cls_score(x)
        proposal_deltas = self.bbox_pred(x)
        return scores, proposal_deltas


class DisAlignCosineFastRCNNOutputLayers(nn.Module):
    """
    Two linear layers for predicting Fast R-CNN outputs:
      (1) proposal-to-detection box regression deltas
      (2) classification scores
    """

    def __init__(
        self,
        input_size,
        num_classes,
        cls_agnostic_bbox_reg,
        box_dim=4,
        scale_mode='learn',
        scale_init=20.0
    ):
        """
        Args:
            input_size (int): channels, or (channels, height, width)
            num_classes (int): number of foreground classes
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            box_dim (int): the dimension of bounding boxes.
                Example box dimensions: 4 for regular XYXY boxes and 5 for rotated XYWHA boxes
        """
        super(DisAlignCosineFastRCNNOutputLayers, self).__init__()

        if not isinstance(input_size, int):
            input_size = np.prod(input_size)

        # The prediction layer for num_classes foreground classes and one background class
        # (hence + 1)
        self.cls_score = NormalizedLinear(
            input_size,
            num_classes + 1,
            scale_mode=scale_mode,
            scale_init=scale_init)
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        self.bbox_pred = nn.Linear(input_size, num_bbox_reg_classes * box_dim)

        # Magnitude and Margin of DisAlign
        self.logit_scale = nn.Parameter(torch.ones(num_classes))
        self.logit_bias = nn.Parameter(torch.zeros(num_classes))
        # Confidence function
        self.confidence_layer = nn.Linear(input_size, 1)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.confidence_layer.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for layer in [self.cls_score, self.confidence_layer, self.bbox_pred]:
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        scores = self.cls_score(x)

        # only adjust the foreground classification scores
        confidence = self.confidence_layer(x).sigmoid()
        scores_tmp = confidence * (scores[:, :-1] * self.logit_scale + self.logit_bias)
        scores_tmp = scores_tmp + (1 - confidence) * scores[:, :-1]

        aligned_scores = cat([scores_tmp, scores[:, -1].view(-1, 1)], dim=1)
        proposal_deltas = self.bbox_pred(x)
        return aligned_scores, proposal_deltas
