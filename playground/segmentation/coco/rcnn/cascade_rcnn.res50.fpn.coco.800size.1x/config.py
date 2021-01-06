import os.path as osp

from cvpods.configs.rcnn_fpn_config import RCNNFPNConfig

_config_dict = dict(
    MODEL=dict(
        WEIGHTS="detectron2://ImageNetPretrained/MSRA/R-50.pkl",
        MASK_ON=True,
        RESNETS=dict(DEPTH=50),
        RPN=dict(
            PRE_NMS_TOPK_TRAIN=2000,
            PRE_NMS_TOPK_TEST=1000,
            POST_NMS_TOPK_TRAIN=2000,
            POST_NMS_TOPK_TEST=1000,
        ),
        ROI_HEADS=dict(
            IN_FEATURES=["p2", "p3", "p4", "p5"], ),
        ROI_BOX_HEAD=dict(
            NUM_FC=2,
            POOLER_RESOLUTION=7,
            CLS_AGNOSTIC_BBOX_REG=True,
        ),
        ROI_BOX_CASCADE_HEAD=dict(
            BBOX_REG_WEIGHTS=(
                (10.0, 10.0, 5.0, 5.0),
                (20.0, 20.0, 10.0, 10.0),
                (30.0, 30.0, 15.0, 15.0),
            ),
            IOUS=(0.5, 0.6, 0.7),
        ),
        ROI_MASK_HEAD=dict(
            NUM_CONV=4,
            POOLER_RESOLUTION=14,
        ),
    ),
    DATASETS=dict(
        TRAIN=("coco_2017_train",),
        TEST=("coco_2017_val",),
    ),
    SOLVER=dict(
        LR_SCHEDULER=dict(
            STEPS=(60000, 80000),
            MAX_ITER=90000,
        ),
        OPTIMIZER=dict(
            BASE_LR=0.02,
        ),
        IMS_PER_BATCH=16,
    ),
    INPUT=dict(
        AUG=dict(
            TRAIN_PIPELINES=[
                ("ResizeShortestEdge",
                 dict(short_edge_length=(800,), max_size=1333, sample_style="choice")),
                ("RandomFlip", dict()),
            ],
            TEST_PIPELINES=[
                ("ResizeShortestEdge",
                 dict(short_edge_length=800, max_size=1333, sample_style="choice")),
            ],
        )
    ),
    OUTPUT_DIR=osp.join(
        '/data/Outputs/model_logs/cvpods_playground',
        osp.split(osp.realpath(__file__))[0].split("playground/")[-1]
    ),
)


class CascadeRCNNConfig(RCNNFPNConfig):
    def __init__(self):
        super(CascadeRCNNConfig, self).__init__()
        self._register_configuration(_config_dict)


config = CascadeRCNNConfig()
