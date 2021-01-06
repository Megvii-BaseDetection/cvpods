import os.path as osp

from cvpods.configs.ssd_config import SSDConfig

_config_dict = dict(
    MODEL=dict(
        WEIGHTS="cvpods/ImageNetPretrained/mmlab/vgg16.pth",
        SSD=dict(
            IMAGE_SIZE=512,
            FEATURE_MAP_SIZE=[64, 32, 16, 8, 4, 2, 1],
            DEFAULT_BOX=dict(
                SCALE=dict(
                    CONV4_3_SCALE=0.04,
                    S_MIN=0.1,
                    S_MAX=0.9,
                ),
                ASPECT_RATIOS=[[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],
                CLIP=False,
            ),
        ),
    ),
    DATASETS=dict(
        TRAIN=("coco_2017_train",),
        TEST=("coco_2017_val",),
    ),
    SOLVER=dict(
        LR_SCHEDULER=dict(
            STEPS=(160000, 180000),
            MAX_ITER=200000,
        ),
        OPTIMIZER=dict(
            BASE_LR=2e-3,
            WEIGHT_DECAY=5e-4,
        ),
        IMS_PER_BATCH=64,
        IMS_PER_DEVICE=8,
    ),
    INPUT=dict(
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
                ("RandomSwapChannels", dict(
                    prob=0.5,
                )),
                ("MinIoURandomCrop", dict(
                    min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.3)),
                ("Resize", dict(shape=(512, 512))),
                ("RandomFlip", dict()),
            ],
            TEST_PIPELINES=[
                ("Resize", dict(shape=(512, 512))),
            ],
        ),
        FORMAT="RGB",
    ),
    TEST=dict(
        EVAL_PERIOD=10000,
    ),
    OUTPUT_DIR=osp.join(
        '/data/Outputs/model_logs/cvpods_playground',
        osp.split(osp.realpath(__file__))[0].split("playground/")[-1]
    ),
)


class CustomSSDConfig(SSDConfig):
    def __init__(self):
        super(CustomSSDConfig, self).__init__()
        self._register_configuration(_config_dict)


config = CustomSSDConfig()
