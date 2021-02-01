import os.path as osp

from cvpods.configs.rcnn_fpn_config import RCNNFPNConfig

_config_dict = dict(
    MODEL=dict(
        WEIGHTS="detectron2://ImageNetPretrained/MSRA/R-50.pkl",
        KEYPOINT_ON=True,
        RESNETS=dict(DEPTH=50),
        RPN=dict(POST_NMS_TOPK_TRAIN=1500, ),
        ROI_HEADS=dict(NUM_CLASSES=1, ),
        ROI_BOX_HEAD=dict(SMOOTH_L1_BETA=0.5, ),
        ROI_KEYPOINT_HEAD=dict(
            NAME="KRCNNConvDeconvUpsampleHead",
            POOLER_RESOLUTION=14,
            POOLER_SAMPLING_RATIO=0,
            CONV_DIMS=tuple(512 for _ in range(8)),
            NUM_KEYPOINTS=17,  # 17 is the number of keypoints in COCO.
            MIN_KEYPOINTS_PER_IMAGE=1,
            NORMALIZE_LOSS_BY_VISIBLE_KEYPOINTS=True,
            LOSS_WEIGHT=1.0,
            POOLER_TYPE="ROIAlignV2",
        ),
    ),
    DATASETS=dict(
        TRAIN=("coco_person_keypoints_2017_train", ),
        TEST=("coco_person_keypoints_2017_val", ),
    ),
    DATALOADER=dict(NUM_WORKERS=4, ),
    SOLVER=dict(
        LR_SCHEDULER=dict(
            STEPS=(60000, 80000),
            MAX_ITER=90000,
        ),
        OPTIMIZER=dict(
            BASE_LR=0.02,
            WEIGHT_DECAY=1e-4,
            WEIGHT_DECAY_NORM=0.0,
        ),
        IMS_PER_BATCH=16,
        CHECKPOINT_PERIOD=5000,
    ),
    INPUT=dict(AUG=dict(
        TRAIN_PIPELINES=[
            ("ResizeShortestEdge",
             dict(short_edge_length=(640, 672, 704, 736, 768, 800),
                  max_size=1333,
                  sample_style="choice")),
            ("RandomFlip", dict()),
        ],
        TEST_PIPELINES=[
            ("ResizeShortestEdge",
             dict(short_edge_length=800, max_size=1333, sample_style="choice")
             ),
        ],
    )),
    TEST=dict(
        EVAL_PERIOD=5000,
    ),
    OUTPUT_DIR=osp.join(
        '/data/Outputs/model_logs/cvpods_playground',
        osp.split(osp.realpath(__file__))[0].split("playground/")[-1]),
)


class KeypointConfig(RCNNFPNConfig):
    def __init__(self):
        super(KeypointConfig, self).__init__()
        self._register_configuration(_config_dict)


config = KeypointConfig()
