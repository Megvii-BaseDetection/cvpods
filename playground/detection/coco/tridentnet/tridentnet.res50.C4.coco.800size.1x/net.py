import sys

from cvpods.layers.shape_spec import ShapeSpec
from cvpods.modeling.backbone import Backbone
from cvpods.modeling.meta_arch.rcnn import GeneralizedRCNN
from cvpods.modeling.roi_heads.box_head import FastRCNNConvFCHead

from tridentnet_base.trident_backbone import build_trident_resnet_backbone
from tridentnet_base.trident_rcnn import TridentRes5ROIHeads
from tridentnet_base.trident_rpn import TridentRPN

sys.path.append("..")


def build_backbone(cfg, input_shape=None):
    if input_shape is None:
        input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))
    backbone = build_trident_resnet_backbone(cfg, input_shape)
    assert isinstance(backbone, Backbone)
    return backbone


def build_proposal_generator(cfg, input_shape):
    return TridentRPN(cfg, input_shape)


def build_roi_heads(cfg, input_shape):
    return TridentRes5ROIHeads(cfg, input_shape)


def build_box_head(cfg, input_shape):
    return FastRCNNConvFCHead(cfg, input_shape)


def build_model(cfg):
    cfg.build_backbone = build_backbone
    cfg.build_proposal_generator = build_proposal_generator
    cfg.build_roi_heads = build_roi_heads
    cfg.build_box_head = build_box_head

    model = GeneralizedRCNN(cfg)
    return model
