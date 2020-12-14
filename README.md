# CVPODS                                                                                                                                                                                                                 

## Introduction

A versatile and efficient codebase for many computer vision tasks: classification, segmentation, detection, self-supervised learning, keypoints and 3D, etc.

### Features

* **Clean & simple & flexible development**: When using detectron2, if you want to implement a new module such as CustomRetinanet, you need to register it to meta_arch, then specify in xxx_config.yaml, and you still need to esplicitly invoke 'from net import CustomRetinanet' to allow registry to retrieve your module successfully. It means you need to copy train_net.py from common tools directory and insert the line above;
* **Flexible and easy-to-use configuration system**: When add new config options in Detectron2, you need to add it into config/defaults.py first and then modify the config.yaml. But now in cvpods you just need to add to config.py once. When you need to debug a component, you may need to set SOLVER.IMS_PER_BATCH to 2, before you need to modify it in config, after it starts running correctly, you need to modify it to 16. That's unconvenient too. So ​cvpods allow you to dynamicly update config, for example: `pods_train --num-gpus 1 SOLVER.IMS_PER_BATCH 2`.
* **Task specific incremental updating.**: For example, if you need to modify Retinanet relative configurations, you just need to modify retinanet_config.py and don't need care other common configs. On the other hand, we avoid putting all kinds of methods' configuration all in one base config file(such as detectron2/config/defaults.py) like detectron2, maskrcnn_benchmark and mmdetection. So retinanet will not include ROI_HEADS, MASK_HEADS configurations, but only has all necessary component.
* **Efficient experiments management**: When you need to implement a new model, you can either copy a project from examples and inheritate some kind of networks such as RetinaNet of FasterRCNN, then define your custom functions; or you can add a new base / commonly used model(such as FCOS) into `cvpods/modeling/meta_arch' and using it like a library.
* **Versatile tasks & datasets support**:
  * Detection, Segmentation (Semantic, Panoptic, Instance), Keypoint, Self-supervised Learning, 3D Detection & Segmentation, etc.
  * COCO, Objects365, WiderFace, VOC, LVIS, CityPersons, ImageNet, CrowdHuman, CityScapes, ModelNet40, ScanNet, KITTI, nuScenes, etc.
* **Global training / testing scripts.**: you just need to invoke `pods_train/test --num-gpus x` in your playground; and your projects only need to include all project-specific configs and network modules.
* **Compatible with detectron2**: All models in detectron2 can be easily migrated into cvpods.

## News

* 2020.07 cvpods is released.

## Requirements

* CUDA 10.1 & cuDNN 7.6.3 & nccl 2.4.8 (optional)
* Python >= 3.6
* PyTorch >= 1.3
* torchvision >= 0.4.2
* OpenCV
* pycocotools
* GCC >= 4.9

## Get Started

```shell

# Install cvpods (requires GPU) locally
python -m pip install 'git+https://github.com/Megvii-BaseDetection/cvpods.git'
# (add --user if you don't have permission)

# Or, to install it from a local clone:
git clone https://github.com/Megvii-BaseDetection/cvpods.git
python -m pip install -e cvpods 

# Or,
pip install -r requirements.txt
python setup.py build develop

# Preprare data path
ln -s /path/to/your/coco/dataset datasets/coco

# Enter a specific experiment dir 
cd playground/retinanet/retinanet.res50.fpn.coco.multiscale.1x

# Train
pods_train --num-gpus 8
# Test
pods_test --num-gpus 8 \
    MODEL.WEIGHTS /path/to/your/save_dir/ckpt.pth # optional
    OUTPUT_DIR /path/to/your/save_dir # optional

# Multi node training
## sudo apt install net-tools ifconfig
pods_train --num-gpus 8 --num-machines N --machine-rank 0/1/.../N-1 --dist-url "tcp://MASTER_IP:port"
```

## Model ZOO

For all the models supported by cvpods, please refer to [MODEL_ZOO](https://github.com/Megvii-BaseDetection/cvpods/blob/master/playground/README.md).

## Projects based on cvpods

* [AutoAssign](https://github.com/Megvii-BaseDetection/AutoAssign)
* [BorderDet](https://github.com/Megvii-BaseDetection/BorderDet)
* [DeFCN](https://github.com/Megvii-BaseDetection/DeFCN)
* [DynamicHead](https://github.com/StevenGrove/DynamicHead)
* [DynamicRouting](https://github.com/Megvii-BaseDetection/DynamicRouting)
* [LearnableTreeFilterV2](https://github.com/StevenGrove/LearnableTreeFilterV2)
* [SelfSup](https://github.com/poodarchu/SelfSup)


## Acknowledgement

cvpods is developed based on Detectron2. For more details about official detectron2, please check [DETECTRON2](https://github.com/facebookresearch/detectron2/blob/master/README.md)


