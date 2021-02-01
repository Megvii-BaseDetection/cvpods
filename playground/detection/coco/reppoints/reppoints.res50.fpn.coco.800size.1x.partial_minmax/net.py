import logging

from cvpods.layers import ShapeSpec
from cvpods.modeling.anchor_generator import ShiftGenerator
from cvpods.modeling.backbone import Backbone
from cvpods.modeling.backbone.fpn import build_retinanet_resnet_fpn_backbone
from cvpods.modeling.meta_arch import RepPoints


def build_backbone(cfg, input_shape=None):
    """
    Build a backbone from `cfg.MODEL.BACKBONE.NAME`.

    Returns:
        an instance of :class:`Backbone`
    """
    if input_shape is None:
        input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))

    backbone = build_retinanet_resnet_fpn_backbone(cfg, input_shape)
    assert isinstance(backbone, Backbone)
    return backbone


def build_shift_generator(cfg, input_shape):

    return ShiftGenerator(cfg, input_shape)


def build_model(cfg):

    cfg.build_backbone = build_backbone
    cfg.build_shift_generator = build_shift_generator

    model = RepPoints(cfg)
    logger = logging.getLogger(__name__)
    logger.info("Model:\n{}".format(model))
    return model
