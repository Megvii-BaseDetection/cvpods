from cvpods.layers import ShapeSpec
from cvpods.modeling.backbone import Backbone, build_resnet_backbone
from cvpods.modeling.meta_arch import FCNHead, SemanticSegmentor


def build_backbone(cfg, input_shape=None):
    if input_shape is None:
        input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))
    backbone = build_resnet_backbone(cfg, input_shape)
    assert isinstance(backbone, Backbone)
    return backbone


def build_sem_seg_head(cfg, input_shape):
    return FCNHead(cfg, input_shape)


def build_model(cfg):
    cfg.build_backbone = build_backbone
    cfg.build_sem_seg_head = build_sem_seg_head

    model = SemanticSegmentor(cfg)
    return model
