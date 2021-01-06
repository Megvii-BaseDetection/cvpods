import os.path as osp

from cvpods.configs.segm_config import SegmentationConfig

_config_dict = dict(
    MODEL=dict(
        WEIGHTS="detectron2://ImageNetPretrained/MSRA/R-50.pkl",
        RESNETS=dict(
            OUT_FEATURES=["res2", "res3", "res4", "res5"],
            DEPTH=50),
        FPN=dict(
            IN_FEATURES=["res2", "res3", "res4", "res5"],
        )
    ),
    DATASETS=dict(
        TRAIN=("coco_2017_train_panoptic_stuffonly",),
        TEST=("coco_2017_val_panoptic_stuffonly",),
    ),
    SOLVER=dict(
        LR_SCHEDULER=dict(
            STEPS=(60000, 80000),
            MAX_ITER=90000,
        ),
        OPTIMIZER=dict(
            BASE_LR=0.02,
        ),
    ),
    INPUT=dict(
        AUG=dict(
            TRAIN_PIPELINES=[
                ("ResizeShortestEdge", dict(
                    short_edge_length=(640, 672, 704, 736, 768, 800),
                    max_size=1333, sample_style="choice")),
                ("RandomFlip", dict()),
            ],
            TEST_PIPELINES=[
                ("ResizeShortestEdge", dict(
                    short_edge_length=800, max_size=1333, sample_style="choice")),
            ],
        ),
    ),
    OUTPUT_DIR=osp.join(
        '/data/Outputs/model_logs/cvpods_playground',
        osp.split(osp.realpath(__file__))[0].split("playground/")[-1]
    ),
)


class SemanticFPNConfig(SegmentationConfig):
    def __init__(self):
        super(SemanticFPNConfig, self).__init__()
        self._register_configuration(_config_dict)


config = SemanticFPNConfig()
