import os.path as osp

from cvpods.configs.solo_config import SOLOConfig

_config_dict = dict(
    MODEL=dict(
        WEIGHTS="detectron2://ImageNetPretrained/MSRA/R-50.pkl",
        SOLO=dict(HEAD=dict(TYPE="DecoupledSOLOHead"))
    ),
    DATASETS=dict(
        TRAIN=("coco_2017_train",),
        TEST=("coco_2017_val",),
    ),
    SOLVER=dict(
        LR_SCHEDULER=dict(
            NAME="WarmupMultiStepLR",
            MAX_ITER=90000,
            STEPS=(60000, 80000),
            WARMUP_FACTOR=1.0 / 1000,
            WARMUP_ITERS=500,
            WARMUP_METHOD="linear",
            GAMMA=0.1,
        ),
        OPTIMIZER=dict(
            NAME="SGD",
            BASE_LR=0.01,
            WEIGHT_DECAY=0.0001,
            MOMENTUM=0.9,
        ),
        CHECKPOINT_PERIOD=5000,
        IMS_PER_BATCH=16,
        IMS_PER_DEVICE=2,
        BATCH_SUBDIVISIONS=1,
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


class CustomSOLOConfig(SOLOConfig):
    def __init__(self):
        super(CustomSOLOConfig, self).__init__()
        self._register_configuration(_config_dict)


config = CustomSOLOConfig()
