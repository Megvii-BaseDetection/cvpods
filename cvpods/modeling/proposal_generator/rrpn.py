# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Dict

import torch

from cvpods.layers import ShapeSpec

from ..anchor_generator import RotatedAnchorGenerator
from ..box_regression import Box2BoxTransformRotated
from .rpn import RPN, StandardRPNHead
from .rrpn_outputs import RRPNOutputs, find_top_rrpn_proposals


class RRPN(RPN):
    """
    Rotated RPN subnetwork.
    Please refer to https://arxiv.org/pdf/1703.01086.pdf for the original RRPN paper:
    Ma, J., Shao, W., Ye, H., Wang, L., Wang, H., Zheng, Y., & Xue, X. (2018).
    Arbitrary-oriented scene text detection via rotation proposals.
    IEEE Transactions on Multimedia, 20(11), 3111-3122.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__(cfg, input_shape)
        self.box2box_transform = Box2BoxTransformRotated(weights=cfg.MODEL.RPN.BBOX_REG_WEIGHTS)
        self.anchor_generator = RotatedAnchorGenerator(
            cfg, [input_shape[f] for f in self.in_features]
        )
        self.rpn_head = StandardRPNHead(cfg, self.anchor_generator,
                                        [input_shape[f] for f in self.in_features])

    def forward(self, images, features, gt_instances=None):
        # same signature as RPN.forward
        gt_boxes = [x.gt_boxes for x in gt_instances] if gt_instances is not None else None
        del gt_instances
        features = [features[f] for f in self.in_features]
        pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features)
        anchors = self.anchor_generator(features)

        outputs = RRPNOutputs(
            self.box2box_transform,
            self.anchor_matcher,
            self.batch_size_per_image,
            self.positive_fraction,
            images,
            pred_objectness_logits,
            pred_anchor_deltas,
            anchors,
            self.boundary_threshold,
            gt_boxes,
            self.smooth_l1_beta,
        )
        if self.training:
            losses = outputs.losses()
        else:
            losses = {}

        with torch.no_grad():
            # Find the top proposals by applying NMS and removing boxes that
            # are too small. The proposals are treated as fixed for approximate
            # joint training with roi heads. This approach ignores the derivative
            # w.r.t. the proposal boxes’ coordinates that are also network
            # responses, so is approximate.
            proposals = find_top_rrpn_proposals(
                outputs.predict_proposals(),
                outputs.predict_objectness_logits(),
                images,
                self.nms_thresh,
                self.pre_nms_topk[self.training],
                self.post_nms_topk[self.training],
                self.min_box_side_len,
            )

        return proposals, losses
