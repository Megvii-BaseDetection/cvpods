import logging

from cvpods.layers import ShapeSpec
from cvpods.modeling.anchor_generator import DefaultAnchorGenerator
from cvpods.modeling.backbone import Backbone, build_efficientnet_bifpn_backbone
from cvpods.modeling.meta_arch import EfficientDet
from cvpods.modeling.nn_utils.parameter_count import parameter_count


def build_backbone(cfg, input_shape=None):
    """
    Build a backbone from `cfg.MODEL.BACKBONE.NAME`.

    Returns:
        an instance of :class:`Backbone`
    """
    if input_shape is None:
        input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))

    backbone = build_efficientnet_bifpn_backbone(cfg, input_shape)
    assert isinstance(backbone, Backbone)
    return backbone


def build_anchor_generator(cfg, input_shape):

    return DefaultAnchorGenerator(cfg, input_shape)


def build_model(cfg):
    cfg.build_backbone = build_backbone
    cfg.build_anchor_generator = build_anchor_generator

    model = EfficientDet(cfg)
    logger = logging.getLogger(__name__)
    logger.info("Model:\n{}".format(model))
    # parameter count
    parameter_count_result = parameter_count(model=model)
    parameter_count_dict = dict(
        EfficientNet=parameter_count_result["backbone.bottom_up"],
        BiFPN=parameter_count_result["backbone"],
        EfficientDet=parameter_count_result[""]
    )
    log_str = "\n".join([
        " => {}: {:.7} M".format(name, count * 1e-6)
        for name, count in parameter_count_dict.items()
    ])
    logger.info("Model #Params:\n{}".format(log_str))
    return model
