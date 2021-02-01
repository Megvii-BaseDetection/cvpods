from cvpods.layers import ShapeSpec
from cvpods.modeling.backbone import Backbone, build_resnet_backbone
from cvpods.modeling.meta_arch import CenterNet


def build_backbone(cfg, input_shape=None):
    """
    Build a backbone.

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
    model = CenterNet(cfg)
    return model
