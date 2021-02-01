from cvpods.layers import ShapeSpec
from cvpods.modeling.backbone import Backbone
from cvpods.modeling.backbone.fpn import build_resnet_fpn_backbone
from cvpods.modeling.meta_arch import SemanticSegmentor, SemSegFPNHead
from cvpods.modeling.meta_arch.pointrend import PointRendSemSegHead
from cvpods.modeling.roi_heads.point_head import StandardPointHead


def build_backbone(cfg, input_shape=None):
    if input_shape is None:
        input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))
    backbone = build_resnet_fpn_backbone(cfg, input_shape)
    assert isinstance(backbone, Backbone)
    return backbone


def build_coarse_sem_seg_head(cfg, input_shape):
    return SemSegFPNHead(cfg, input_shape)


def build_sem_seg_head(cfg, input_shape):
    return PointRendSemSegHead(cfg, input_shape)


def build_point_head(cfg, input_shape):
    return StandardPointHead(cfg, input_shape)


def build_model(cfg):
    cfg.build_backbone = build_backbone
    cfg.build_coarse_sem_seg_head = build_coarse_sem_seg_head
    cfg.build_sem_seg_head = build_sem_seg_head
    cfg.build_point_head = build_point_head

    model = SemanticSegmentor(cfg)
    return model
