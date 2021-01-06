from cvpods.layers import ShapeSpec
from cvpods.modeling.backbone import Backbone
from cvpods.modeling.backbone.fpn import build_resnet_fpn_backbone
from cvpods.modeling.meta_arch.solo import SOLO
from cvpods.modeling.meta_arch.solo_decoupled import DecoupledSOLO


def build_backbone(cfg, input_shape=None):
    if input_shape is None:
        input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))
    backbone = build_resnet_fpn_backbone(cfg, input_shape)
    assert isinstance(backbone, Backbone)
    return backbone


def build_model(cfg):
    cfg.build_backbone = build_backbone
    solo_head = cfg.MODEL.SOLO.HEAD.TYPE
    if solo_head in ["SOLOHead"]:
        model = SOLO(cfg)
    elif solo_head in ["DecoupledSOLOHead"]:
        model = DecoupledSOLO(cfg)
    else:
        raise ValueError(
            f"Unknow SOLO head type: {solo_head}. "
            "Please select from ['SOLOHead', 'DecoupledSOLOHead'].")
    return model
