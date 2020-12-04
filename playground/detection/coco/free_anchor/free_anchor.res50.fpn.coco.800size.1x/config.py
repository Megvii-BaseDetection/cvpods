import os.path as osp

from cvpods.configs.retinanet_config import RetinaNetConfig

_config_dict = dict(
    MODEL=dict(
        WEIGHTS="detectron2://ImageNetPretrained/MSRA/R-50.pkl",
        RESNETS=dict(DEPTH=50),
        RETINANET=dict(
            IOU_THRESHOLDS=[0.4, 0.5],
            IOU_LABELS=[0, -1, 1],
            PRIOR_PROB=0.02,
            NMS_THRESH_TEST=0.5,
            BBOX_REG_WEIGHTS=(10.0, 10.0, 5.0, 5.0),
            FOCAL_LOSS_GAMMA=2.0,
            FOCAL_LOSS_ALPHA=0.5,
            SMOOTH_L1_LOSS_BETA=0.1,
            REG_WEIGHT=0.75,
        ),
        FREE_ANCHOR=dict(
            POS_ANCHOR_TOPK=50,
            BBOX_THRESHOLD=0.6,
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


class FreeAnchorConfig(RetinaNetConfig):
    def __init__(self):
        super(FreeAnchorConfig, self).__init__()
        self._register_configuration(_config_dict)


config = FreeAnchorConfig()
