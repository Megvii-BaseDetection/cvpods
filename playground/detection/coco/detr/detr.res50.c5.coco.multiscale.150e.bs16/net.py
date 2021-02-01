import logging
import sys

from cvpods.layers import ShapeSpec
from cvpods.modeling.backbone import Backbone, build_resnet_backbone
from cvpods.modeling.meta_arch.detr import DETR

from optimizer import OPTIMIZER_BUILDER  # noqa

sys.path.append("..")


def build_backbone(cfg, input_shape=None):
    """
    Build a backbone from `cfg.MODEL.BACKBONE.NAME`.

    Returns:
        an instance of :class:`Backbone`
    """
    if input_shape is None:
        input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))

    backbone = build_resnet_backbone(cfg, input_shape)
    assert isinstance(backbone, Backbone)
    return backbone


def build_model(cfg):
    cfg.build_backbone = build_backbone
    model = DETR(cfg)

    logger = logging.getLogger(__name__)
    logger.info("Model:\n{}".format(model))
    return model
