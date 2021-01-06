import os.path as osp

from cvpods.configs.rcnn_fpn_config import RCNNFPNConfig

_config_dict = dict(
    MODEL=dict(
        WEIGHTS="cvpods/ImageNetPretrained/torchvision/resnet50.pth",
        PIXEL_MEAN=[123.675, 116.280, 103.530],
        PIXEL_STD=[58.395, 57.120, 57.375],
        MASK_ON=False,
        RESNETS=dict(
            DEPTH=50,
            OUT_FEATURES=["res2", "res3", "res4", "res5"],
            STRIDE_IN_1X1=False,
        ),
        FPN=dict(IN_FEATURES=["res2", "res3", "res4", "res5"]),
        ROI_HEADS=dict(IN_FEATURES=["p2", "p3", "p4", "p5"]),
        ROI_BOX_HEAD=dict(
            POOLER_TYPE="ROIAlignV2",
            POOLER_RESOLUTION=7,
            POOLER_SAMPLING_RATIO=2,
        ),
        SPARSE_RCNN=dict(
            NUM_CLASSES=80,
            NUM_PROPOSALS=100,
            # RCNN Head
            DROPOUT=0.0,
            DIM_FEEDFORWARD=2048,
            ACTIVATION='relu',
            HIDDEN_DIM=256,
            NHEADS=8,
            NUM_CLS=1,
            NUM_REG=3,
            NUM_HEADS=6,
            # D-Conv
            NUM_DYNAMIC=2,
            DIM_DYNAMIC=64,
            # loss
            CLASS_WEIGHT=2.0,
            GIOU_WEIGHT=2.0,
            L1_WEIGHT=5.0,
            DEEP_SUPERVISION=True,
            NO_OBJECT_WEIGHT=0.1,
            # focal loss
            USE_FOCAL=True,
            ALPHA=0.25,
            GAMMA=2.0,
            PRIOR_PROB=0.01,
        ),
    ),
    DATASETS=dict(
        TRAIN=("coco_2017_train",),
        TEST=("coco_2017_val",),
    ),
    SOLVER=dict(
        LR_SCHEDULER=dict(
            STEPS=(210000, 250000),
            MAX_ITER=270000,
            WARMUP_FACTOR=0.01,
            WARMUP_ITERS=1000
        ),
        OPTIMIZER=dict(
            NAME="FullModelAdamWBuilder",
            BASE_LR=0.000025,
            BASE_LR_RATIO_BACKBONE=1.,
            BETAS=(0.9, 0.999),
            EPS=1e-08,
            AMSGRAD=False,
        ),
        CLIP_GRADIENTS=dict(
            CLIP_TYPE="norm",
            CLIP_VALUE=1.0,
            ENABLED=True,
            FULL_MODEL=True,
            NORM_TYPE=2.0,
        ),
        IMS_PER_BATCH=16,
    ),
    SEED=40244023,
    INPUT=dict(
        AUG=dict(
            TRAIN_PIPELINES=[
                ("ResizeShortestEdge", dict(
                    short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
                    max_size=1333, sample_style="choice")),
                ("RandomFlip", dict()),
            ],
            TEST_PIPELINES=[
                ("ResizeShortestEdge",
                 dict(short_edge_length=800, max_size=1333, sample_style="choice")),
            ],
        ),
        FORMAT="RGB",
    ),
    TEST=dict(
        EVAL_PERIOD=45000,
    ),
    OUTPUT_DIR=osp.join(
        '/data/Outputs/model_logs/cvpods_playground',
        osp.split(osp.realpath(__file__))[0].split("playground/")[-1]),
)


class SparseRCNNConfig(RCNNFPNConfig):
    def __init__(self):
        super(SparseRCNNConfig, self).__init__()
        self._register_configuration(_config_dict)


config = SparseRCNNConfig()
