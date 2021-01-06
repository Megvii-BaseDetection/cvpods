import torch.nn

from cvpods.modeling.backbone import build_darknet_backbone
from cvpods.modeling.meta_arch import YOLOv3


def build_model(cfg):

    cfg.build_backbone = build_darknet_backbone
    model = YOLOv3(cfg)
    # YOLOv3 can apply official SyncBN
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    return model
