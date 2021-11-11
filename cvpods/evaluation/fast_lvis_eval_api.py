#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2019-2021 Megvii Inc. All rights reserved.

import copy
import time

import numpy as np
from lvis import LVISEval
from loguru import logger

from cvpods import _C


class LVISEval_opt(LVISEval):
    """
    This is a slightly modified version of the original LVIS API, where the functions evaluate_img()
    and accumulate() are implemented in C++ to speedup evaluation
    """

    def evaluate(self):
        """
        Run per image evaluation on given images and store results
        (a list of dict) in self.eval_imgs.
        """
        logger.info("Running per image evaluation.")
        logger.info("Evaluate annotation type *{}*".format(self.params.iou_type))
        tic = time.time()

        self.params.img_ids = list(np.unique(self.params.img_ids))

        if self.params.use_cats:
            cat_ids = self.params.cat_ids
        else:
            cat_ids = [-1]

        self._prepare()

        self.ious = {
            (img_id, cat_id): self.compute_iou(img_id, cat_id)
            for img_id in self.params.img_ids
            for cat_id in cat_ids
        }

        # <<<< Beginning of code differences with original LVIS API
        p = self.params

        def convert_instances_to_cpp(instances, neg_list, is_det=False):
            # Convert annotations for a list of instances in an image to a format that's fast
            # to access in C++
            instances_cpp = []
            for instance in instances:
                instance_cpp = _C.LVISInstanceAnnotation(
                    int(instance["id"]),
                    instance["score"] if is_det else instance.get("score", 0.0),
                    instance["area"],
                    True if instance['category_id'] in neg_list else False,
                    bool(instance.get("iscrowd", 0)),
                    bool(instance.get("ignore", 0)),
                )
                instances_cpp.append(instance_cpp)
            return instances_cpp

        # Convert GT annotations, detections, and IOUs to a format that's fast to access in C++
        ground_truth_instances = [
            [convert_instances_to_cpp(self._gts[imgId, catId], self.img_nel[imgId])
             for catId in p.cat_ids] for imgId in p.img_ids
        ]
        detected_instances = [
            [convert_instances_to_cpp(self._dts[imgId, catId], self.img_nel[imgId], is_det=True)
             for catId in p.cat_ids] for imgId in p.img_ids
        ]
        ious = [[self.ious[imgId, catId] for catId in cat_ids] for imgId in p.img_ids]

        if not p.use_cats:
            # For each image, flatten per-category lists into a single list
            ground_truth_instances = [[[o for c in i for o in c]] for i in ground_truth_instances]
            detected_instances = [[[o for c in i for o in c]] for i in detected_instances]

        # Call C++ implementation of self.evaluateImgs()
        self._eval_imgs_cpp = _C.LVISevalEvaluateImages(
            p.area_rng, p.max_dets, p.iou_thrs, ious, ground_truth_instances, detected_instances
        )
        self._eval_imgs = None

        toc = time.time()
        logger.info("LVISeval_opt.evaluate() finished in {:0.2f} seconds.".format(toc - tic))

    def accumulate(self):
        """Accumulate per image evaluation results and store the result in
        self.eval.
        """
        logger.info("Accumulating evaluation results.")
        tic = time.time()

        if not hasattr(self, "_eval_imgs_cpp"):
            logger.error("Please run evaluate() first")

        self._params_eval = copy.deepcopy(self.params)
        source = ["rec_thrs", "max_dets", "iou_thrs", "rec_thrs", "use_cats",
                  "cat_ids", "area_rng", "max_dets", "img_ids"]
        target = ["recThrs", "maxDets", "iouThrs", "recThrs", "useCats",
                  "catIds", "areaRng", "maxDets", "imgIds"]
        for s, t in zip(source, target):
            self._params_eval.__setattr__(t, self._params_eval.__getattribute__(s))
        self._params_eval.maxDets = [self._params_eval.maxDets]

        self.eval = _C.LVISevalAccumulate(self._params_eval, self._eval_imgs_cpp)

        # recall is num_iou_thresholds X num_categories X num_area_ranges X num_max_detections
        self.eval["recall"] = np.array(self.eval["recall"]).reshape(
            self.eval["counts"][:1] + self.eval["counts"][2:]
        )

        # precision and scores are num_iou_thresholds X num_recall_thresholds X num_categories X
        # num_area_ranges X num_max_detections
        self.eval["precision"] = np.array(self.eval["precision"]).reshape(self.eval["counts"])
        self.eval["scores"] = np.array(self.eval["scores"]).reshape(self.eval["counts"])

        toc = time.time()
        logger.info("LVISeval_opt.accumulate() finished in {:0.2f} seconds.".format(toc - tic))
