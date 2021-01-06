import os.path as osp

from cvpods.configs.retinanet_config import RetinaNetConfig

_config_dict = dict(
    MODEL=dict(
        WEIGHTS="detectron2://ImageNetPretrained/MSRA/R-50.pkl",
        RESNETS=dict(DEPTH=50),
        RETINANET=dict(
            NUM_CLASSES=1,
            IOU_THRESHOLDS=[0.4, 0.6],
            IOU_LABELS=[0, -1, 1],
            NMS_THRESH_TEST=0.5,
            BBOX_REG_WEIGHTS=(1.0, 1.0, 1.0, 1.0),
            FOCAL_LOSS_GAMMA=2.0,
            FOCAL_LOSS_ALPHA=0.25,
            SMOOTH_L1_LOSS_BETA=0.1,
        ),
    ),
    DATASETS=dict(
        TRAIN=("citypersons_train",),
        TEST=("citypersons_val",),
    ),
    SOLVER=dict(
        LR_SCHEDULER=dict(
            NAME="WarmupMultiStepLR",
            STEPS=(12000, 16000),
            MAX_ITER=18000,
            WARMUP_FACTOR=1.0 / 4000,
            WARMUP_ITERS=4000,
            WARMUP_METHOD="linear",
            GAMMA=0.1,
        ),
        IMS_PER_BATCH=16,
        OPTIMIZER=dict(
            BASE_LR=0.01,
        ),
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
    ),
    OUTPUT_DIR=osp.join(
        '/data/Outputs/model_logs/cvpods_playground',
        osp.split(osp.realpath(__file__))[0].split("playground/")[-1]
    ),
)


class CustomRetinaNetConfig(RetinaNetConfig):
    def __init__(self):
        super(CustomRetinaNetConfig, self).__init__()
        self._register_configuration(_config_dict)


config = CustomRetinaNetConfig()
