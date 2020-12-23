- [cvpods_playground](#cvpods_playground)
  * [ImageNet Classification](#imagenet-classification)
  * [Self\-supervised Learning](#self--supervised-learning)
  * [Object Detection](#object-detection)
    + [COCO](#coco)
      - [Faster R-CNN](#faster-r-cnn)
      - [RetinaNet](#retinanet)
      - [FCOS](#fcos)
      - [ATSS](#atss)
      - [FreeAnchor](#freeanchor)
      - [TridentNet](#tridentnet)
      - [RepPoints](#reppoints)
      - [DETR](#detr)
    + [PASCAL VOC](#pascal-voc)
    + [WIDER FACE](#wider-face)
    + [CityPersons](#citypersons)
    + [CrowdHuman](#crowdhuman)
  * [Instance Segmentation](#instance-segmentation)
  * [Semantic Segmentation](#semantic-segmentation)
  * [Panoptic Segmentation](#panoptic-segmentation)
  * [Key\-Points](#key-points)
    + [COCO_PERSON](#coco_person)
      - [Keypoint\-RCNN](#keypoint-rcnn)
  * [3D](#3D)

# Model Zoo

> All experiments are conducted on servers with 8 NVIDIA V100 / 2080Ti GPUs (PCIE). The software in use were PyTorch 1.3, CUDA 10.1, cuDNN 7.6.3.

## ImageNet Classification

Comming Soon.

## Self\-supervised Learning

Comming Soon.

## Object Detection 

### COCO

#### Faster R-CNN

| Name                                                         | input size | lr sched | train time (s/iter) | train mem (GB) | box AP |
| ------------------------------------------------------------ | ---------- | -------- | ------------------- | -------------- | ------ |
| [FasterRCNN-R50-FPN](playground/detection/coco/rcnn/faster_rcnn.res50.fpn.coco.multiscale.1x) | 640-800    | 90k      | 0.225(2080ti)       | 2.82           | 38.1   |
| [FasterRCNN-R50-FPN-SyncBN](playground/detection/coco/rcnn/faster_rcnn.res50.fpn.coco.multiscale.2x.syncbn) | 640-800    | 180k     | 0.546               | 5.23           | 39.9   |

#### RetinaNet

| Name                                                         | input size | lr sched | train time (s/iter) | train mem (GB) | box AP |
| ------------------------------------------------------------ | ---------- | -------- | ------------------- | -------------- | ------ |
| [RetinaNet-R50](playground/detection/coco/retinanet/retinanet.res50.fpn.coco.800size.1x) | 800        | 90k      | 0.3593              | 3.85           | 35.9 |
| [RetinaNet-R50](playground/detection/coco/retinanet/retinanet.res50.fpn.coco.multiscale.1x) | 640-800    | 90k      | 0.244               | 3.84           | 36.5 |

#### FCOS

| Name                                                         | input size | lr sched | train time (s/iter) | train mem (GB) | box AP |
| ------------------------------------------------------------ | ---------- | -------- | ------------------- | -------------- |   --- |
| [FCOS-R50-FPN](playground/detection/coco/fcos/fcos.res50.fpn.coco.800size.1x) | 800        | 90k      | 0.334(2080ti)       | 3.09           | 38.8  |


#### ATSS

| Name                                                         | input size | lr sched | train time (s/iter) | train mem (GB) | box AP |
| ------------------------------------------------------------ | ---------- | -------- | ------------------- | -------------- |   --- |
| [ATSS-R50-FPN](playground/detection/coco/atss/atss.res50.fpn.coco.800size.1x) | 800        | 90k      | 0.340(2080ti)       | 3.09           | 39.3  |

#### FreeAnchor

| Name                                                         | input size | lr sched | train time (s/iter) | train mem (GB) | box AP |
| ------------------------------------------------------------ | ---------- | -------- | ------------------- | -------------- | --- |
| [FreeAnchor-R50-FPN](playground/detection/coco/free_anchor/free_anchor.res50.fpn.coco.800size.1x) | 800        | 90k      | 0.353(2080ti)       | 4.08           | 38.3  |

#### TridentNet

| Name                                                         | input size | lr sched | train time (s/iter) | train mem (GB) | box AP |
| ------------------------------------------------------------ | ---------- | -------- | ------------------- | -------------- | --- |
| [TridentNet-R50-C4](playground/detection/coco/tridentnet/tridentnet.res50.C4.coco.800size.1x) | 800        | 90k      | 0.754(2080ti)       | 4.65           | 37.7  |

#### RepPoints

| Name                                                         | input size | lr sched | train time (s/iter) | train mem (GB) | box AP |
| ------------------------------------------------------------ | ---------- | -------- | ------------------- | -------------- | --- |
| [RepPoints-R50-FPN](playground/detection/coco/reppoints/reppoints.res50.fpn.coco.800size.1x.partial_minmax) | 800        | 90k      | 0.415(2080ti)       | 2.85           | 38.2  |

#### DETR

| Named                                                        | input size | lr sched | train time (s/iter) | train mem (GB) | box AP |
| ------------------------------------------------------------ | ---------- | -------- | ------------------- | -------------- | --- |
| [DETR-R50-C5](playground/detection/coco/detr/detr.res50.c5.coco.multiscale.150e.bs16) | 480-800    | 150e     | 0.270(v100)         | 3.62           | 38.7   |


### PASCAL VOC

Comming Soon.

### WIDER FACE

Comming Soon.

### CityPersons

Comming Soon.

### CrowdHuman

Comming Soon.

## Instance Segmentation 

 Comming Soon.

## Semantic Segmentation

Comming Soon.

## Panoptic Segmentation

Comming Soon.

# Key-Points

## COCO_PERSON

### Keypoint-RCNN

| Named                                                        | input size | lr sched | train time (s/iter) | train mem (GB) |  box AP |  kp AP |
| ------------------------------------------------------------ | ---------- | -------- | ------------------- | -------------- | ------ |  ------ |
| [RCNN_R50_FPN](playground/keypoints/coco_person/rcnn/keypoint_rcnn.res50.FPN.coco_person.multiscale.1x) | 480-800    | 90k     | 0.4r0(2080Ti)       | 4.47      | 53.7 | 64.2 | 

# 3D

Comming Soon.
