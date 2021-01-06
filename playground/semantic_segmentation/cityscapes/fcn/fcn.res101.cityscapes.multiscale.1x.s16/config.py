import os.path as osp

from cvpods.configs.segm_config import SegmentationConfig

_config_dict = dict(
    MODEL=dict(
        WEIGHTS="detectron2://ImageNetPretrained/MSRA/R-101.pkl",
        RESNETS=dict(
            DEPTH=101,
            OUT_FEATURES=["res4", "res5"],
            FREEZE_AT=0,
        ),
        SEM_SEG_HEAD=dict(
            NUM_CLASSES=19,
            IN_FEATURES=["res4", "res5"],
            IGNORE_VALUE=255,
            LOSS_WEIGHT=1.0,
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
    INPUT=dict(
        AUG=dict(
            TRAIN_PIPELINES=[
                ("ResizeShortestEdge", dict(
                    short_edge_length=(512, 768, 1024, 1280, 1536, 1792, 2048),
                    max_size=4096, sample_style="choice")),
                ("RandomCropPad", dict(
                    crop_type="absolute", crop_size=(800, 800), img_value=0, seg_value=255)),
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


class FCNConfig(SegmentationConfig):
    def __init__(self):
        super(FCNConfig, self).__init__()
        self._register_configuration(_config_dict)


config = FCNConfig()
