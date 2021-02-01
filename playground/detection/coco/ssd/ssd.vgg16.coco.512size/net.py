import logging

from cvpods.layers import ShapeSpec
from cvpods.modeling.backbone import Backbone
from cvpods.modeling.backbone.vgg import build_ssd_vgg_backbone
from cvpods.modeling.meta_arch import SSD
from cvpods.modeling.meta_arch.ssd import DefaultBox


def build_backbone(cfg, input_shape=None):
    """
    Build a backbone from `cfg.MODEL.BACKBONE.NAME`.

    Returns:
        an instance of :class:`Backbone`
    """
    if input_shape is None:
        input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))

    backbone = build_ssd_vgg_backbone(cfg, input_shape)
    assert isinstance(backbone, Backbone)
    return backbone


def build_default_box_generator(cfg):
    return DefaultBox(cfg)


def build_model(cfg):
    cfg.build_backbone = build_backbone
    cfg.build_default_box_generator = build_default_box_generator

    model = SSD(cfg)
    logger = logging.getLogger(__name__)
    logger.info("Model:\n{}".format(model))
    return model
