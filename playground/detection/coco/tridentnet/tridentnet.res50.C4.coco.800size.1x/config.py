import os.path as osp

from cvpods.configs.rcnn_config import RCNNConfig

_config_dict = dict(
    MODEL=dict(
        WEIGHTS="detectron2://ImageNetPretrained/MSRA/R-50.pkl",
        MASK_ON=False,
        RESNETS=dict(
            DEPTH=50,
            OUT_FEATURES=["res4"],
        ),
        ANCHOR_GENERATOR=dict(
            SIZES=[[32, 64, 128, 256, 512]],
        ),
        RPN=dict(
            IN_FEATURES=["res4"],
            PRE_NMS_TOPK_TRAIN=12000,
            PRE_NMS_TOPK_TEST=6000,
            POST_NMS_TOPK_TRAIN=500,
            POST_NMS_TOPK_TEST=1000,
        ),
        ROI_HEADS=dict(
            POSITIVE_FRACTION=0.5,
            BATCH_SIZE_PER_IMAGE=128,
            PROPOSAL_APPEND_GT=False,
        ),
        ROI_BOX_HEAD=dict(
            NUM_FC=2,
            POOLER_RESOLUTION=14,
        ),
        TRIDENT=dict(
            NUM_BRANCH=3,
            BRANCH_DILATIONS=[1, 2, 3],
            TEST_BRANCH_IDX=1,
            TRIDENT_STAGE="res4",
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


class TridentNetConfig(RCNNConfig):
    def __init__(self):
        super(TridentNetConfig, self).__init__()
        self._register_configuration(_config_dict)


config = TridentNetConfig()
