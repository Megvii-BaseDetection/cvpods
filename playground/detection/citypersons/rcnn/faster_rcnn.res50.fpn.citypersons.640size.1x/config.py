import os.path as osp

from cvpods.configs.rcnn_fpn_config import RCNNFPNConfig

_config_dict = dict(
    MODEL=dict(
        WEIGHTS="detectron2://ImageNetPretrained/MSRA/R-50.pkl",
        MASK_ON=False,
        RESNETS=dict(DEPTH=50),
        RPN=dict(
            PRE_NMS_TOPK_TRAIN=2000,
            PRE_NMS_TOPK_TEST=1000,
            POST_NMS_TOPK_TRAIN=1000,
            POST_NMS_TOPK_TEST=1000,
        ),
        ROI_HEADS=dict(
            # NAME="StandardROIHeads",
            IN_FEATURES=["p2", "p3", "p4", "p5"],
            NUM_CLASSES=1,
        ),
        ROI_BOX_HEAD=dict(
            # NAME="FastRCNNConvFCHead",
            NUM_FC=2,
            POOLER_RESOLUTION=7,
        ),
        ROI_MASK_HEAD=dict(
            # NAME="MaskRCNNConvUpsampleHead",
            NUM_CONV=4,
            POOLER_RESOLUTION=14,
        ),
    ),
    DATASETS=dict(
        TRAIN=("citypersons_train",),
        TEST=("citypersons_val",),
    ),
    SOLVER=dict(
        LR_SCHEDULER=dict(
            MAX_ITER=9000,
            STEPS=(6000, 8000),
        ),
        OPTIMIZER=dict(
            BASE_LR=0.02,
        ),
        CHECKPOINT_PERIOD=1000,
        IMS_PER_BATCH=16,
    ),
    INPUT=dict(
        AUG=dict(
            TRAIN_PIPELINES=[
                ("RandomCropWithInstance", dict(
                    crop_type="relative_range", crop_size=[0.3, 0.3])),
                ("ResizeShortestEdge", dict(
                    short_edge_length=(640,), max_size=1280, sample_style="choice")),
                ("RandomFlip", dict()),
            ],
            TEST_PIPELINES=[
            ],
        ),
        CROP=dict(ENABLED=True, TYPE="relative_range", SIZE=[0.3, 0.3],),
    ),
    OUTPUT_DIR=osp.join(
        '/data/Outputs/model_logs/cvpods_playground',
        osp.split(osp.realpath(__file__))[0].split("playground/")[-1]
    ),
)


class FasterRCNNConfig(RCNNFPNConfig):
    def __init__(self):
        super(FasterRCNNConfig, self).__init__()
        self._register_configuration(_config_dict)


config = FasterRCNNConfig()
