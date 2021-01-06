import os.path as osp

from cvpods.configs.retinanet_config import RetinaNetConfig

_config_dict = dict(
    MODEL=dict(
        WEIGHTS="detectron2://ImageNetPretrained/MSRA/R-50.pkl",
        RESNETS=dict(DEPTH=50),
        RETINANET=dict(
            NUM_CLASSES=1,
            IOU_THRESHOLDS=[0.4, 0.5],
            IOU_LABELS=[0, -1, 1],
            TOPK_CANDIDATES_TEST=5000,
            NMS_THRESH_TEST=0.5,
            BBOX_REG_WEIGHTS=(1.0, 1.0, 1.0, 1.0),
            FOCAL_LOSS_GAMMA=2.0,
            FOCAL_LOSS_ALPHA=0.25,
            SMOOTH_L1_LOSS_BETA=0.1,
        ),
    ),
    DATASETS=dict(
        TRAIN=("widerface_2019_train",),
        TEST=("widerface_2019_val",),
    ),
    SOLVER=dict(
        LR_SCHEDULER=dict(
            MAX_ITER=45000,
            STEPS=(30000, 40000),
        ),
        OPTIMIZER=dict(
            BASE_LR=0.01,
        ),
        CHECKPOINT_PERIOD=2500,
        IMS_PER_BATCH=16,
    ),
    INPUT=dict(
        AUG=dict(
            TRAIN_PIPELINES=[
                ("RandomCropWithInstance", dict(
                    crop_type="relative_range", crop_size=(0.25, 0.25))),
                ("ResizeShortestEdge", dict(
                    short_edge_length=(600,), max_size=1500, sample_style="choice")),
                ("ShuffleList", dict(transforms=[
                    ("RandomBrightness", dict(intensity_min=0.6, intensity_max=1.4)),
                    ("RandomContrast", dict(intensity_min=0.6, intensity_max=1.4)),
                    ("RandomSaturation", dict(intensity_min=0.6, intensity_max=1.4)),
                    ("RandomLighting", dict(scale=0.1)),
                ])),
                ("RandomFlip", dict()),
            ],
            TEST_PIPELINES=[
                ("ResizeShortestEdge", dict(
                    short_edge_length=1000, max_size=2500, sample_style="choice")),
            ],
        ),
    ),
    TEST=dict(
        DETECTIONS_PER_IMAGE=1000,
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
