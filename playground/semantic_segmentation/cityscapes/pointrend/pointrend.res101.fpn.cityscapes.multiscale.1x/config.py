import os.path as osp

from cvpods.configs.pointrend_config import PointRendRCNNFPNConfig

_config_dict = dict(
    MODEL=dict(
        WEIGHTS="detectron2://ImageNetPretrained/MSRA/R-101.pkl",
        RESNETS=dict(
            DEPTH=101,
            FREEZE_AT=0,
        ),
        SEM_SEG_HEAD=dict(
            NUM_CLASSES=19,
            IN_FEATURES=["p2", "p3", "p4", "p5"],
            IGNORE_VALUE=255,
            NORM="GN",
            COMMON_STRIDE=4,
            CONVS_DIM=128,
            LOSS_WEIGHT=1.0,
        ),
        POINT_HEAD=dict(
            NUM_CLASSES=19,
            FC_DIM=256,
            NUM_FC=3,
            IN_FEATURES=["p2"],
            TRAIN_NUM_POINTS=2048,
            SUBDIVISION_STEPS=2,
            SUBDIVISION_NUM_POINTS=8192,
        ),
    ),
    DATASETS=dict(
        TRAIN=("cityscapes_fine_sem_seg_train",),
        TEST=("cityscapes_fine_sem_seg_val",),
    ),
    SOLVER=dict(
        LR_SCHEDULER=dict(
            STEPS=(40000, 55000),
            MAX_ITER=65000,
        ),
        OPTIMIZER=dict(
            BASE_LR=0.01,
        ),
        IMS_PER_BATCH=32,
        IMS_PER_DEVICE=4,
    ),
    DATALOADER=dict(
        NUM_WORKERS=16,
    ),
    INPUT=dict(
        AUG=dict(
            TRAIN_PIPELINES=[
                ("ResizeShortestEdge", dict(
                    short_edge_length=(512, 768, 1024, 1280, 1536, 1792, 2048),
                    max_size=4096, sample_style="choice")),
                # SSD Color Aug List
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
                ("RandomDistortion", dict(
                    hue=18.0 / 255, saturation=1., exposure=1., image_format="BGR"
                )),
                ("RandomCropWithMaxAreaLimit", dict(
                    crop_type="absolute", crop_size=(512, 1024), single_category_max_area=0.75)),
                ("RandomFlip", dict()),
            ],
            TEST_PIPELINES=[
                ("ResizeShortestEdge", dict(
                    short_edge_length=1024, max_size=2048, sample_style="choice")),
            ],
        ),
    ),
    OUTPUT_DIR=osp.join(
        '/data/Outputs/model_logs/cvpods_playground',
        osp.split(osp.realpath(__file__))[0].split("playground/")[-1]
    ),
)


class CustomPointRendRCNNFPNConfig(PointRendRCNNFPNConfig):
    def __init__(self):
        super(CustomPointRendRCNNFPNConfig, self).__init__()
        self._register_configuration(_config_dict)


config = CustomPointRendRCNNFPNConfig()
