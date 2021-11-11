#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates. All Rights Reserved

import logging
import os
import os.path as osp
from copy import deepcopy
from loguru import logger

import numpy as np

import torch

from cvpods.utils import Timer

from ..base_dataset import BaseDataset
from ..registry import DATASETS
from .paths_route import _PREDEFINED_SPLITS_IMAGENETLT
from .imagenet_categories import IMAGENET_CATEGORIES

"""
This file contains functions to parse ImageNet-format annotations into dicts in "cvpods format".
"""


@DATASETS.register()
class ImageNetLTDataset(BaseDataset):
    def __init__(self, cfg, dataset_name, transforms=[], is_train=True):
        super(ImageNetLTDataset, self).__init__(cfg, dataset_name, transforms, is_train)

        image_root, label_file = _PREDEFINED_SPLITS_IMAGENETLT["imagenetlt"][self.name]

        self.label_file = osp.join(self.data_root, label_file)
        self.image_root = osp.join(self.data_root, image_root)

        self.meta = self._get_metadata()
        self.dataset_dicts = self._load_annotations()

        self._set_group_flag()
        self.eval_with_gt = cfg.TEST.get("WITH_GT", False)

    def __getitem__(self, index):
        """Load data, apply transforms, converto to Instances.
        """
        dataset_dict = deepcopy(self.dataset_dicts[index])

        # read image
        image = self._read_data(dataset_dict["file_name"])

        annotations = dataset_dict.get("annotations", None)

        # apply transfrom
        images, annotations = self._apply_transforms(
            image, annotations)

        def process(dd, img, annos):
            if isinstance(annos, list):
                annos = [a for a in annos if a is not None]

            # image shape: CHW / NCHW
            # TODO: fix hack
            if img.shape[0] == 3:  # CHW
                dd["image"] = torch.as_tensor(np.ascontiguousarray(img))
            elif len(img.shape) == 3 and img.shape[-1] == 3:
                dd["image"] = torch.as_tensor(
                    np.ascontiguousarray(img.transpose(2, 0, 1)))
            elif len(img.shape) == 4 and img.shape[-1] == 3:
                # NHWC -> NCHW
                dd["image"] = torch.as_tensor(
                    np.ascontiguousarray(img.transpose(0, 3, 1, 2)))
            return dd

        if isinstance(images, dict):
            ret = {}
            # multiple input pipelines
            for desc, item in images.items():
                img, anno = item
                ret[desc] = process(deepcopy(dataset_dict), img, anno)
            return ret
        else:
            return process(dataset_dict, images, annotations)

    def __len__(self):
        return len(self.dataset_dicts)

    def _get_metadata(self):
        assert len(IMAGENET_CATEGORIES.keys()) == 1000
        cat_ids = [v[0] for v in IMAGENET_CATEGORIES.values()]
        assert min(cat_ids) == 1 and max(cat_ids) == len(cat_ids), \
            "Category ids are not in [1, #categories], as expected"
        # Ensure that the category list is sroted by id
        imagenet_categories = sorted(IMAGENET_CATEGORIES.items(), key=lambda x: x[1][0])
        thing_classes = [v[1][1] for v in imagenet_categories]
        meta = {
            "thing_classes": thing_classes,
            "evaluator_type": _PREDEFINED_SPLITS_IMAGENETLT["evaluator_type"]["imagenetlt"],
        }
        return meta

    def _load_annotations(self):
        timer = Timer()

        """Constructs the imdb."""
        # Compile the split data path
        logger.info('{} data path: {}'.format(self.name, self.label_file))

        # Construct the image db
        imdb = []
        f = open(self.label_file, "r")
        for line in f.readlines():
            img_path, label = line.strip().split(" ")
            imdb.append({
                "im_path": os.path.join(self.image_root, img_path),
                "class": int(label),
            })
        f.close()

        logging.info("Loading {} takes {:.2f} seconds.".format(self.label_file, timer.seconds()))

        dataset_dicts = []
        for i, item in enumerate(imdb):
            dataset_dicts.append({
                "image_id": i,
                "category_id": item["class"],
                "file_name": item["im_path"],
            })

        return dataset_dicts
