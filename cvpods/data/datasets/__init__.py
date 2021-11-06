#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2019-2021 Megvii Inc. All rights reserved.

from .citypersons import CityPersonsDataset
from .cityscapes import CityScapesDataset
from .coco import COCODataset
from .crowdhuman import CrowdHumanDataset
from .imagenet import ImageNetDataset
from .imagenetlt import ImageNetLTDataset
from .lvis import LVISDataset
from .objects365 import Objects365Dataset
from .torchvision_datasets import CIFAR10Dataset, STL10Datasets
from .voc import VOCDataset
from .widerface import WiderFaceDataset

__all__ = [k for k in globals().keys() if not k.startswith("_")]
