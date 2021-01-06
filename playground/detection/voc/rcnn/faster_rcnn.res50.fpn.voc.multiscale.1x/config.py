import os.path as osp

from cvpods.configs.rcnn_fpn_config import RCNNFPNConfig

_config_dict = dict(
    MODEL=dict(
        WEIGHTS="detectron2://ImageNetPretrained/MSRA/R-50.pkl",
        MASK_ON=False,
        RESNETS=dict(DEPTH=50),
        ROI_HEADS=dict(NUM_CLASSES=20),
    ),
    DATASETS=dict(
        TRAIN=("voc_2007_trainval", "voc_2012_trainval"),
        TEST=("voc_2007_test",),
    ),
    SOLVER=dict(
        IMS_PER_BATCH=16,
        CHECKPOINT_PERIOD=3000,
        LR_SCHEDULER=dict(
            STEPS=(12000, 16000),
            MAX_ITER=18000,
            WARMUP_ITERS=100,
        ),
        OPTIMIZER=dict(
            BASE_LR=0.02,
        ),
    ),
    INPUT=dict(
        AUG=dict(
            TRAIN_PIPELINES=[
                ("ResizeShortestEdge",
                 dict(short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
                      max_size=1333, sample_style="choice")),
                ("RandomFlip", dict()),
            ],
            TEST_PIPELINES=[
                ("ResizeShortestEdge",
                 dict(short_edge_length=800, max_size=1333, sample_style="choice")),
            ],
        )
    ),
    TEST=dict(
        EVAL_PERIOD=3000,
    ),
    OUTPUT_DIR=osp.join(
        '/data/Outputs/model_logs/cvpods_playground',
        osp.split(osp.realpath(__file__))[0].split("playground/")[-1]),
)


class FasterRCNNConfig(RCNNFPNConfig):
    def __init__(self):
        super(FasterRCNNConfig, self).__init__()
        self._register_configuration(_config_dict)


config = FasterRCNNConfig()
