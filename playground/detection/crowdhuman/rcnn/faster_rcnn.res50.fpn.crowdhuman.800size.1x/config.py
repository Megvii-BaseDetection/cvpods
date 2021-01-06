import os.path as osp

from cvpods.configs.rcnn_fpn_config import RCNNFPNConfig

_config_dict = dict(
    MODEL=dict(
        WEIGHTS="detectron2://ImageNetPretrained/MSRA/R-50.pkl",
        MASK_ON=False,
        RESNETS=dict(DEPTH=50),
        ANCHOR_GENERATOR=dict(
            SIZES=[[32], [64], [128], [256], [512]], ASPECT_RATIOS=[[1.0, 2.0, 3.0]],
        ),
        PROPOSAL_GENERATOR=dict(
            MIN_SIZE=2,
        ),
        RPN=dict(
            PRE_NMS_TOPK_TRAIN=12000,
            PRE_NMS_TOPK_TEST=6000,
            POST_NMS_TOPK_TRAIN=2000,
            POST_NMS_TOPK_TEST=1000,
            SMOOTH_L1_BETA=1.0,
        ),
        ROI_HEADS=dict(
            # NAME="StandardROIHeads",
            IN_FEATURES=["p2", "p3", "p4", "p5"],
            POSITIVE_FRACTION=0.5,
            NUM_CLASSES=1,
        ),
        ROI_BOX_HEAD=dict(
            # NAME="FastRCNNConvFCHead",
            NUM_FC=2,
            POOLER_RESOLUTION=7,
            SMOOTH_L1_BETA=1.0,
        ),
        ROI_MASK_HEAD=dict(
            # NAME="MaskRCNNConvUpsampleHead",
            NUM_CONV=4,
            POOLER_RESOLUTION=14,
        ),
    ),
    DATASETS=dict(
        TRAIN=("crowdhuman_train",),
        TEST=("crowdhuman_val",),
    ),
    SOLVER=dict(
        LR_SCHEDULER=dict(
            MAX_ITER=28125,
            STEPS=(18750, 24375),
        ),
        OPTIMIZER=dict(
            BASE_LR=0.02,
        ),
        CHECKPOINT_PERIOD=1000,
        IMS_PER_BATCH=16,
    ),
    TEST=dict(
        # Maximum number of detections to return per image during inference (100 is
        # based on the limit established for the COCO dataset).
        DETECTIONS_PER_IMAGE=300,
    ),
    INPUT=dict(
        AUG=dict(
            TRAIN_PIPELINES=[
                ("ResizeShortestEdge",
                 dict(short_edge_length=(800,), max_size=1400, sample_style="choice")),
                ("RandomFlip", dict()),
            ],
            TEST_PIPELINES=[
                ("ResizeShortestEdge",
                 dict(short_edge_length=800, max_size=1400, sample_style="choice")),
            ],
        )
    ),
    OUTPUT_DIR=osp.join(
        '/data/Outputs/model_logs/cvpods_playground',
        osp.split(osp.realpath(__file__))[0].split("playground/")[-1]
    ),
)


class FasterRCNNConfig(RCNNFPNConfig):
    def __init__(self):
        super(FasterRCNNConfig, self).__init__()
        self._register_configuration(_config_dict)


config = FasterRCNNConfig()
