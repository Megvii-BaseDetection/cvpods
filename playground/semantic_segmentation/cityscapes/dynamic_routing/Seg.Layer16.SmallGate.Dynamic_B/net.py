import logging

from cvpods.layers import ShapeSpec
from cvpods.modeling.backbone import Backbone
from cvpods.modeling.backbone.dynamic_arch.dynamic_backbone import build_dynamic_backbone
from cvpods.modeling.meta_arch.dynamic4seg import DynamicNet4Seg, SemSegDecoderHead


def build_backbone(cfg, input_shape=None):
    """
    Build a backbone from `cfg.MODEL.BACKBONE.NAME`.

    Returns:
        an instance of :class:`Backbone`
    """
    if input_shape is None:
        input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN),
                                height=cfg.INPUT.FIX_SIZE_FOR_FLOPS[0],
                                width=cfg.INPUT.FIX_SIZE_FOR_FLOPS[1])

    backbone = build_dynamic_backbone(cfg, input_shape)
    assert isinstance(backbone, Backbone)
    return backbone


def build_sem_seg_head(cfg, input_shape=None):
    return SemSegDecoderHead(cfg, input_shape)


def build_model(cfg):
    cfg.build_backbone = build_backbone
    cfg.build_sem_seg_head = build_sem_seg_head
    model = DynamicNet4Seg(cfg)
    logger = logging.getLogger(__name__)
    logger.info("Model:\n{}".format(model))
    return model
