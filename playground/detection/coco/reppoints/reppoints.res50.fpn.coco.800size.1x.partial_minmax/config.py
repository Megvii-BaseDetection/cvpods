import os.path as osp

from cvpods.configs.base_detection_config import BaseDetectionConfig

_config_dict = dict(
    MODEL=dict(
        WEIGHTS="detectron2://ImageNetPretrained/MSRA/R-50.pkl",
        RESNETS=dict(
            DEPTH=50,
            OUT_FEATURES=["res3", "res4", "res5"],
        ),
        FPN=dict(
            IN_FEATURES=["res3", "res4", "res5"],
            NORM="GN",
        ),
        SHIFT_GENERATOR=dict(
            NUM_SHIFTS=1,
            OFFSET=0,
        ),
        REPPOINTS=dict(
            NUM_CLASSES=80,
            IN_FEATURES=["p3", "p4", "p5", "p6", "p7"],
            FPN_STRIDES=[8, 16, 32, 64, 128],
            POINT_BASE_SCALE=4,
            FEAT_CHANNELS=256,
            POINT_FEAT_CHANNELS=256,
            STACK_CONVS=3,
            NORM_MODE='GN',
            PRIOR_PROB=0.01,
            SCORE_THRESH_TEST=0.05,
            TOPK_CANDIDATES_TEST=1000,
            NMS_THRESH_TEST=0.5,
            FOCAL_LOSS_GAMMA=2.0,
            FOCAL_LOSS_ALPHA=0.25,
            LOSS_CLS_WEIGHT=1.0,
            LOSS_BBOX_INIT_WEIGHT=0.5,
            LOSS_BBOX_REFINE_WEIGHT=1.0,
            NUM_POINTS=9,
            TRANSFORM_METHOD='partial_minmax',
            GRADIENT_MUL=0.1,
            MOMENT_MUL=0.01,
        ),
    ),
    DATASETS=dict(
        TRAIN=("coco_2017_train", ),
        TEST=("coco_2017_val", ),
    ),
    SOLVER=dict(
        LR_SCHEDULER=dict(
            STEPS=(60000, 80000),
            MAX_ITER=90000,
        ),
        OPTIMIZER=dict(
            BASE_LR=0.01,
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


class RepPointsConfig(BaseDetectionConfig):
    def __init__(self):
        super(RepPointsConfig, self).__init__()
        self._register_configuration(_config_dict)


config = RepPointsConfig()
