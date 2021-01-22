
# Benchmark and Model Zoo

## Environment

### Hardware

- 8 NVIDIA 2080Ti GPUs
- Intel Xeon 4114 CPU @ 2.20GHz

### Software environment

- Python 3.6 / 3.7
- PyTorch 1.3
- CUDA 10.1.243
- CUDNN 7.6.3

## Mirror sites

We use AWS as the main site to host our model zoo.  
You can access hhb oss site `s3://wangfengdata/cvpods_modelzoo` to get all the model list below

## Common settings

- All models use 800 short size by default
- All models were trained on `coco_2017_train`, and tested on the `coco_2017_val`.
- We use distributed training and BN layer stats are fixed.
- All pytorch-style pretrained backbones on ImageNet are from PyTorch model zoo.


## Future works

- More models with different backbones will be added to the model zoo in the future.
- More detection models will be added in model zoo
- Other task models will be added in model zoo

### Detection

#### Faster R-CNN

|    Backbone     |  Style  | Lr schd | Mem (GB) | Train time (s/iter) | Inf time (s/im) | box AP |                                                              Link                                                              |
| :-------------: | :-----: | :-----: | :------: | :-----------------: | :------------: | :----: | :--------------------------------------------------------------------------------------------------------------------------------: |
|     R-50-C4     | pytorch |   1x    |    -     |          -          |      0.122     |  35.7  |         [code](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_r50_c4_1x-642cf91f.pth)          |
|    R-50-FPN     | pytorch |   1x    |    -     |          -          |        -       |    -   |    [code](https://git-core.megvii-inc.com/zhubenjin/cvpods_playground/tree/master/examples/rcnn/faster_rcnn.res50.fpn.coco.800size.1x)     |

#### Mask R-CNN

|    Backbone     |  Style  | Lr schd | Mem (GB) | Train time (s/iter) | Inf time (s/im) | box AP | mask AP |                                                             Link                                                             |
| :-------------: | :-----: | :-----: | :------: | :-----------------: | :------------: | :----: | :-----: | :------------------------------------------------------------------------------------------------------------------------------: |
|    R-50-FPN     | pytorch |   1x    |   3.0    |          -          |      0.090     |  38.2  |  34.8   |    [code](https://git-core.megvii-inc.com/zhubenjin/cvpods_playground/tree/master/examples/rcnn/mask_rcnn.res50.fpn.coco.800size.1x)     |

#### Cascade R-CNN

|    Backbone     |  Style  | Lr schd | Mem (GB) | Train time (s/iter) | Inf time (s/im) | box AP | mask AP |                                                                 Link                                                                  |
| :-------------: | :-----: | :-----: | :------: | :-----------------: | :------------: | :----: | :-----: | :---------------------------------------------------------------------------------------------------------------------------------------: |
|    R-50-FPN     | pytorch |   1x    |   3.4    |          -          |      0.106     |  41.6  |   36.0  |     [code](https://git-core.megvii-inc.com/zhubenjin/cvpods_playground/tree/master/examples/rcnn/cascade_rcnn.res50.fpn.coco.800size.1x)     |

#### TridentNet

|    Backbone     |  Style  | Lr schd | Mem (GB) | Train time (s/iter) | Inf time (s/im) | box AP | mask AP |                                                                 Link                                                                  |
| :-------------: | :-----: | :-----: | :------: | :-----------------: | :------------: | :----: | :-----: | :---------------------------------------------------------------------------------------------------------------------------------------: |
|    R-50-C4      | pytorch |   1x    |   4.8    |          -          |      0.137     |  37.8  |    -    |     [code](https://git-core.megvii-inc.com/zhubenjin/cvpods_playground/tree/master/examples/TridentNet/tridentnet.res50.C4.coco.800size.1x)     |

#### RetinaNet

|    Backbone     |  Style  | Lr schd | Mem (GB) | Train time (s/iter) | Inf time (s/im) | box AP |                                                             Link                                                             |
| :-------------: | :-----: | :-----: | :------: | :-----------------: | :------------: | :----: | :------------------------------------------------------------------------------------------------------------------------------: |
|    R-50-FPN     | pytorch |   1x    |     -    |           -         |         -      |  36.3  |    [code](https://git-core.megvii-inc.com/zhubenjin/cvpods_playground/tree/master/examples/retinanet/retinanet.res50.fpn.coco.800size.1x)     |

#### FCOS

|    Backbone     |  Style  | Lr schd | Mem (GB) | Train time (s/iter) | Inf time (s/im) | box AP |                                                             Link                                                             |
| :-------------: | :-----: | :-----: | :------: | :-----------------: | :------------: | :----: | :------------------------------------------------------------------------------------------------------------------------------: |
|    R-50-FPN     | pytorch |   1x    |    -     |           -         |         -      |  37.0  |    [code](https://git-core.megvii-inc.com/zhubenjin/cvpods_playground/tree/master/examples/fcos/fcos.res50.fpn.coco.800size.1x)     |

#### FreeAnchor

|    Backbone     |  Style  | Lr schd | Mem (GB) | Train time (s/iter) | Inf time (s/im) | box AP |                                                                 Link                                                                  |
| :-------------: | :-----: | :-----: | :------: | :-----------------: | :------------:  | :----: | :---------------------------------------------------------------------------------------------------------------------------------------: |
|    R-50-FPN     | pytorch |   1x    |    -     |           -         |         -       |  38.6  |     [code](https://git-core.megvii-inc.com/zhubenjin/cvpods_playground/tree/master/examples/retinanet/free_anchor.res50.fpn.coco.800size.1x)     |

#### RepPoints

|    Backbone     |  Style  | convert_funv | Lr schd | Mem (GB) | Train time (s/iter) | Inf time (s/im) | box AP |                                                                Link                         |
| :-------------: | :-----: | :-----------: | :-----: | :------: | :-----------------: | :------------:  | :----: | :----------------------------------------------------------------------------------------: |
|    R-50-FPN     | pytorch |     minmax     | 1x           |    2913     |           0.41         |         60.8       |  37.3  |     code     |
|    R-50-FPN     | pytorch | partial_minmax | 1x           |    2914     |           0.40         |         62.4       |  38.1  |     code     |
|    R-50-FPN     | pytorch |     moment     | 1x           |    2914     |           0.40         |         60.9       |  38.0  |     code     |
|    R-50-FPN     | pytorch |     moment     | 2x           |    2916     |           -            |         -          |        |     code     |
|    R-50-FPN     | pytorch |     moment     | 2x (ms_train)|    3377     |           0.45         |         63.2       |  40.2  |     code     |
|    R-101-FPN    | pytorch |     moment     | 2x (ms_train)|    4687     |           0.59         |         80.5       |  42.2  |     code     |

#### TensorMask 

|    Backbone     |  Style  | Lr schd | Mem (GB) | Train time (s/iter) | Inf time (s/im) | box AP | segm AP |                                                                 Link                            |
| :-------------: | :-----: | :-----: | :------: | :-----------------: | :------------:  | :----: | :----: | :----------------------------------------------------------------------------------------------: |
|    R-50-FPN     | pytorch |   1x    |    -     |           -         |         -       |  37.4  | 32.5 | [code](https://git-core.megvii-inc.com/zhubenjin/cvpods_playground/tree/master/examples/tensormask/tensormask.res50.fpn.coco.800size.1x) |

#### CenterNet

#### CornerNet

### Segmentation

#### PSPNet
#### DeepLab v3+
#### GCNet
#### TreeFilter

### Pose estimation

#### DensePose
|    Backbone     |  Style  | Lr schd | Train size | Mem (GB) | Train time (s/iter) | Inf time (s/im) | box AP | densepose AP |   Link |
| :-------------: | :-----: | :-----: | :------: | :------: | :-----------------: | :------------: | :------------: | :----: | :--------------------------------------------------------------------------------------------------------------------------------------------: |
|     R-50-FPN     | pytorch |   s1x    |   800    |    -     |          -          |     -     |  55.6  |   46.3  |     [code](https://git-core.megvii-inc.com/zhubenjin/cvpods_playground/tree/zhoupenghao/examples/Densepose/densepose.res50.fpn.coco.800size.s1x)  |
|     R-50-FPN     | pytorch |   s1x    |   mst    |    -     |          -          |     -     |  57.6  |   49.8  |  [code](https://git-core.megvii-inc.com/zhubenjin/cvpods_playground/tree/zhoupenghao/examples/Densepose/densepose.res50.fpn.coco.mst.s1x)  |
|     R-101-FPN     | pytorch |   s1x    |   800    |    -     |          -          |     -     |  57.3  |   49.1  |     [code](https://git-core.megvii-inc.com/zhubenjin/cvpods_playground/tree/zhoupenghao/examples/Densepose/densepose.res101.fpn.coco.800size.s1x)  |
|     R-101-FPN     | pytorch |   s1x    |   mst    |    -     |          -          |     -     |  59.5  |   51.2  |     [code](https://git-core.megvii-inc.com/zhubenjin/cvpods_playground/tree/zhoupenghao/examples/Densepose/densepose.res101.fpn.coco.mst.s1x)  |
