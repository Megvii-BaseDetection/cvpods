import os.path as osp

from cvpods.configs.base_detection_config import BaseDetectionConfig

_config_dict = dict(
    MODEL=dict(
        MASK_ON=True,
        WEIGHTS="detectron2://ImageNetPretrained/MSRA/R-50.pkl",
        RESNETS=dict(
            DEPTH=50,
            OUT_FEATURES=["res2", "res3", "res4", "res5"]
        ),
        FPN=dict(
            IN_FEATURES=["res2", "res3", "res4", "res5"],
            FUSE_TYPE="avg",
        ),
        ANCHOR_GENERATOR=dict(
            SIZES=[[44, 60], [88, 120], [176, 240], [
                352, 480], [704, 960], [1408, 1920]],
            ASPECT_RATIOS=[[1.0]],
        ),
        TENSOR_MASK=dict(
            IN_FEATURES=["p2", "p3", "p4", "p5", "p6", "p7"],
            NUM_CONVS=4,
            NUM_CLASSES=80,
            CLS_CHANNELS=256,
            SCORE_THRESH_TEST=0.05,
            TOPK_CANDIDATES_TEST=6000,
            NMS_THRESH_TEST=0.5,
            BBOX_CHANNELS=128,
            BBOX_REG_WEIGHTS=(1.5, 1.5, 0.75, 0.75),
            FOCAL_LOSS_GAMMA=3.0,
            FOCAL_LOSS_ALPHA=0.3,
            MASK_CHANNELS=128,
            MASK_LOSS_WEIGHT=2.0,
            POSITIVE_WEIGHT=1.5,
            ALIGNED_ON=True,
            BIPYRAMID_ON=True,
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


class TensorMaskConfig(BaseDetectionConfig):
    def __init__(self):
        super(TensorMaskConfig, self).__init__()
        self._register_configuration(_config_dict)


config = TensorMaskConfig()
