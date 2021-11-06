#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2019-2021 Megvii Inc. All rights reserved.

import os
import os.path as osp
from copy import deepcopy
import megfile
from loguru import logger

import numpy as np

import torch

from cvpods.utils import Timer

from ..base_dataset import BaseDataset
from ..registry import DATASETS
from .imagenet_categories import IMAGENET_CATEGORIES
from .paths_route import _PREDEFINED_SPLITS_IMAGENET


"""
This file contains functions to parse ImageNet-format annotations into dicts in "cvpods format".
"""


@DATASETS.register()
class ImageNetDataset(BaseDataset):
    def __init__(self, cfg, dataset_name, transforms=[], is_train=True):
        super(ImageNetDataset, self).__init__(cfg, dataset_name, transforms, is_train)

        image_root, label_file = _PREDEFINED_SPLITS_IMAGENET["imagenet"][self.name]
        self.label_file = osp.join(self.data_root, image_root, label_file) \
            if "://" not in image_root else osp.join(image_root, label_file)
        self.image_root = osp.join(self.data_root, image_root) \
            if "://" not in image_root else image_root

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
            "evaluator_type": _PREDEFINED_SPLITS_IMAGENET["evaluator_type"]["imagenet"],
        }
        return meta

    def _load_annotations(self):
        timer = Timer()

        """Constructs the imdb."""
        # Compile the split data path
        logger.info('{} data path: {}'.format(self.name, self.label_file))
        # Images are stored per class in subdirs (format: n<number>)
        class_ids = [k for k, v in IMAGENET_CATEGORIES.items()]
        class_id_cont_id = {k: v[0] - 1 for k, v in IMAGENET_CATEGORIES.items()}
        # class_ids = sorted([
        #     f for f in os.listdir(split_path) if re.match(r'^n[0-9]+$', f)
        # ])
        # # Map ImageNet class ids to contiguous ids
        # class_id_cont_id = {v: i for i, v in enumerate(class_ids)}
        # Construct the image db
        imdb = []
        for class_id in class_ids:
            cont_id = class_id_cont_id[class_id]
            im_dir = megfile.smart_path_join(self.label_file, class_id)
            for im_name in megfile.smart_listdir(im_dir):
                imdb.append({
                    'im_path': os.path.join(im_dir, im_name),
                    'class': cont_id,
                })

        logger.info("Loading {} takes {:.2f} seconds.".format(self.label_file, timer.seconds()))

        dataset_dicts = []
        for i, item in enumerate(imdb):
            dataset_dicts.append({
                "image_id": i,
                "category_id": item["class"],
                "file_name": item["im_path"],
            })

        return dataset_dicts
