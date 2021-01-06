import os.path as osp

from cvpods.configs.yolo_config import YOLO3Config

_config_dict = dict(
    MODEL=dict(
        PIXEL_MEAN=(0.485, 0.456, 0.406),
        PIXEL_STD=(0.229, 0.224, 0.225),
        YOLO=dict(
            CLASSES=80,
            IN_FEATURES=["dark3", "dark4", "dark5"],
            ANCHORS=[
                [[116, 90], [156, 198], [373, 326]],
                [[30, 61], [62, 45], [42, 119]],
                [[10, 13], [16, 30], [33, 23]],
            ],
            CONF_THRESHOLD=0.01,  # TEST
            NMS_THRESHOLD=0.5,
            IGNORE_THRESHOLD=0.7,
        ),
    ),
    DATASETS=dict(
        TRAIN=("coco_2017_train",),
        TEST=("coco_2017_val",),
    ),
    SOLVER=dict(
        LR_SCHEDULER=dict(
            NAME="WarmupMultiStepLR",
            MAX_ITER=500000,
            STEPS=(400000, 470000),
            WARMUP_ITERS=2000,
            WARMUP_METHOD="burnin",
        ),
        OPTIMIZER=dict(
            NAME="SGD",
            BASE_LR=0.001,
            MOMENTUM=0.9,
            WEIGHT_DECAY=0.0005,
            WEIGHT_DECAY_BIAS=0.0005,
            WEIGHT_DECAY_NORM=0.0005,
        ),
        IMS_PER_BATCH=64,
        IMS_PER_DEVICE=8,
    ),
    DATALOADER=dict(
        NUM_WORKERS=4,
    ),
    INPUT=dict(
        MIXUP=False,
        FORMAT="RGB",
        AUG=dict(
            TRAIN_PIPELINES=[
                ("RandomBrightness", dict(
                    intensity_min=1 - 32.0 / 255,
                    intensity_max=1 + 32.0 / 255,
                    prob=0.5,
                )),
                ("RandomContrast", dict(
                    intensity_min=0.5,
                    intensity_max=1.5,
                    prob=0.5,
                )),
                ("RandomSaturation", dict(
                    intensity_min=0.5,
                    intensity_max=1.5,
                    prob=0.5,
                )),
                ('MinIoURandomCrop', dict(
                    min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.3)),
                # ('RandomAffine', dict(prob=0.5, borderValue=(123.675, 116.280, 103.530))),
                ('Expand', dict(
                    ratio_range=(1, 4), mean=[123.675, 116.280, 103.530], prob=0.6)),
                ('Resize', dict(shape=(512, 512))),
                ('RandomFlip', dict()),
            ],
            TEST_PIPELINES=[
                ('Resize', dict(shape=(608, 608))),
            ]
        ),
    ),
    OUTPUT_DIR=osp.join(
        '/data/Outputs/model_logs/cvpods_playground',
        osp.split(osp.realpath(__file__))[0].split("playground/")[-1]
    ),
    GLOBAL=dict(DUMP_TEST=False)
)


class CustomYOLO3Config(YOLO3Config):

    def __init__(self):
        super(CustomYOLO3Config, self).__init__()
        self._register_configuration(_config_dict)


config = CustomYOLO3Config()
