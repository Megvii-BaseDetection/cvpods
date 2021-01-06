import os.path as osp

from cvpods.configs.rcnn_fpn_config import RCNNFPNConfig

_config_dict = dict(
    MODEL=dict(
        WEIGHTS="cvpods/ImageNetPretrained/FAIR/maskrcnn_3x.pkl",
        MASK_ON=True,
        RESNETS=dict(DEPTH=50),
        ROI_HEADS=dict(NUM_CLASSES=8),
    ),
    DATASETS=dict(
        TRAIN=("cityscapes_fine_instance_seg_train",),
        TEST=("cityscapes_fine_instance_seg_val",),
    ),
    SOLVER=dict(
        IMS_PER_BATCH=8,
        LR_SCHEDULER=dict(
            STEPS=(18000,),
            MAX_ITER=24000,
        ),
        OPTIMIZER=dict(
            BASE_LR=0.01,
        ),
    ),
    INPUT=dict(
        AUG=dict(
            TRAIN_PIPELINES=[
                ("ResizeShortestEdge", dict(
                    short_edge_length=(800, 832, 864, 896, 928, 960, 992, 1024),
                    max_size=2048, sample_style="choice")),
                ("RandomFlip", dict()),
            ],
            TEST_PIPELINES=[
                ("ResizeShortestEdge", dict(
                    short_edge_length=1024, max_size=2048, sample_style="choice")),
            ],
        ),
    ),
    OUTPUT_DIR=osp.join(
        '/data/Outputs/model_logs/cvpods_playground',
        osp.split(osp.realpath(__file__))[0].split("playground/")[-1]),
)


class MaskRCNNConfig(RCNNFPNConfig):
    def __init__(self):
        super(MaskRCNNConfig, self).__init__()
        self._register_configuration(_config_dict)


config = MaskRCNNConfig()
