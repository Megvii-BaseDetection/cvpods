import logging

from cvpods.layers import ShapeSpec
from cvpods.modeling.backbone import Backbone
from cvpods.modeling.backbone.fpn import build_resnet_fpn_backbone
from cvpods.modeling.meta_arch import PanopticFPN, SemSegFPNHead
from cvpods.modeling.proposal_generator import RPN
from cvpods.modeling.roi_heads import StandardROIHeads
from cvpods.modeling.roi_heads.box_head import FastRCNNConvFCHead
from cvpods.modeling.roi_heads.mask_head import MaskRCNNConvUpsampleHead


def build_backbone(cfg, input_shape=None):
    if input_shape is None:
        input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))
    backbone = build_resnet_fpn_backbone(cfg, input_shape)
    assert isinstance(backbone, Backbone)
    return backbone


def build_sem_seg_head(cfg, input_shape=None):
    return SemSegFPNHead(cfg, input_shape)


def build_proposal_generator(cfg, input_shape):
    return RPN(cfg, input_shape)


def build_roi_heads(cfg, input_shape):
    return StandardROIHeads(cfg, input_shape)


def build_box_head(cfg, input_shape):
    return FastRCNNConvFCHead(cfg, input_shape)


def build_mask_head(cfg, input_shape):
    return MaskRCNNConvUpsampleHead(cfg, input_shape)


def build_model(cfg):

    cfg.build_backbone = build_backbone
    cfg.build_proposal_generator = build_proposal_generator
    cfg.build_roi_heads = build_roi_heads
    cfg.build_box_head = build_box_head
    cfg.build_mask_head = build_mask_head
    cfg.build_sem_seg_head = build_sem_seg_head

    model = PanopticFPN(cfg)
    logger = logging.getLogger(__name__)
    logger.info("Model:\n{}".format(model))
    return model
