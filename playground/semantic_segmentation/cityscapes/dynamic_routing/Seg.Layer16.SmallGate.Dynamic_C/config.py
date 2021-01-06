import os.path as osp

from cvpods.configs.dynamic_routing_config import SemanticSegmentationConfig

_config_dict = dict(
    MODEL=dict(
        WEIGHTS="",
        CAL_FLOPS=True,
        BACKBONE=dict(
            CELL_TYPE=['sep_conv_3x3', 'skip_connect'],
            LAYER_NUM=16,
            CELL_NUM_LIST=[2, 3, 4] + [4 for _ in range(13)],
            INIT_CHANNEL=64,
            MAX_STRIDE=32,
            SEPT_STEM=True,
            NORM="nnSyncBN",
            DROP_PROB=0.0,
        ),
        GATE=dict(
            GATE_ON=True,
            GATE_INIT_BIAS=1.5,
            SMALL_GATE=True,
        ),
        SEM_SEG_HEAD=dict(
            IN_FEATURES=['layer_0', 'layer_1', 'layer_2', 'layer_3'],
            NUM_CLASSES=19,
            IGNORE_VALUE=255,
            NORM="nnSyncBN",
            LOSS_WEIGHT=1.0,
        ),
        BUDGET=dict(
            CONSTRAIN=True,
            LOSS_WEIGHT=0.5,
            LOSS_MU=0.2,
            FLOPS_ALL=26300.0,
            UNUPDATE_RATE=0.4,
            WARM_UP=True,
        ),
    ),
    DATASETS=dict(
        TRAIN=("cityscapes_fine_sem_seg_train", ),
        TEST=("cityscapes_fine_sem_seg_val", ),
    ),
    SOLVER=dict(
        LR_SCHEDULER=dict(
            NAME="PolyLR",
            POLY_POWER=0.9,
            MAX_ITER=190000,
        ),
        OPTIMIZER=dict(BASE_LR=0.05, ),
        IMS_PER_BATCH=8,
        IMS_PER_DEVICE=2,
        CHECKPOINT_PERIOD=5000,
        GRAD_CLIP=5.0,
    ),
    INPUT=dict(
        AUG=dict(
            TRAIN_PIPELINES=[
                ("ResizeShortestEdge", dict(
                    short_edge_length=(512, 768, 1024, 1280, 1536, 2048, ),
                    max_size=4096, sample_style="choice")),
                ("RandomCropPad", dict(
                    crop_type="absolute", crop_size=(768, 768), img_value=0, seg_value=255)),
                ("RandomFlip", dict()),
            ],
            TEST_PIPELINES=[
                ("ResizeShortestEdge",
                 dict(short_edge_length=1024, max_size=2048, sample_style="choice")),
            ],
        ),
        # FIX_SIZE_FOR_FLOPS=[768, 768],
        FIX_SIZE_FOR_FLOPS=[1024, 2048],
    ),
    OUTPUT_DIR=osp.join(
        '/data/Outputs/model_logs/cvpods_playground',
        osp.split(osp.realpath(__file__))[0].split("playground/")[-1]),
)


class DynamicSemanticSegmentationConfig(SemanticSegmentationConfig):
    def __init__(self):
        super(DynamicSemanticSegmentationConfig, self).__init__()
        self._register_configuration(_config_dict)


config = DynamicSemanticSegmentationConfig()
