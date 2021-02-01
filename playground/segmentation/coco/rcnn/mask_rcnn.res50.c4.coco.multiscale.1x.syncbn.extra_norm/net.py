from cvpods.layers import ShapeSpec, get_norm
from cvpods.modeling.backbone import Backbone
from cvpods.modeling.backbone.resnet import build_resnet_backbone
from cvpods.modeling.meta_arch.rcnn import GeneralizedRCNN
from cvpods.modeling.proposal_generator import RPN
from cvpods.modeling.roi_heads import Res5ROIHeads
from cvpods.modeling.roi_heads.mask_head import MaskRCNNConvUpsampleHead


class Res5ROIHeadsExtraNorm(Res5ROIHeads):
    """
    As described in the MOCO paper, there is an extra BN layer
    following the res5 stage.
    """
    def _build_res5_block(self, cfg):
        seq, out_channels = super()._build_res5_block(cfg)
        norm = cfg.MODEL.RESNETS.NORM
        norm = get_norm(norm, out_channels)
        seq.add_module("norm", norm)
        return seq, out_channels


def build_backbone(cfg, input_shape=None):
    if input_shape is None:
        input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))
    backbone = build_resnet_backbone(cfg, input_shape)
    assert isinstance(backbone, Backbone)
    return backbone


def build_proposal_generator(cfg, input_shape):
    return RPN(cfg, input_shape)


def build_roi_heads(cfg, input_shape):
    return Res5ROIHeadsExtraNorm(cfg, input_shape)


def build_mask_head(cfg, input_shape):
    return MaskRCNNConvUpsampleHead(cfg, input_shape)


def build_model(cfg):
    cfg.build_backbone = build_backbone
    cfg.build_proposal_generator = build_proposal_generator
    cfg.build_roi_heads = build_roi_heads
    cfg.build_mask_head = build_mask_head

    model = GeneralizedRCNN(cfg)
    return model
