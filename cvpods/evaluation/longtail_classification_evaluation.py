#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates. All Rights Reserved

import copy
import itertools
import os
import os.path as osp
from collections import OrderedDict
import json
from sklearn.metrics import f1_score
from loguru import logger

import torch

import cvpods
from cvpods.utils import PathManager, comm, create_small_table

from .evaluator import DatasetEvaluator
from .registry import EVALUATOR


@EVALUATOR.register()
class LongTailClassificationEvaluator(DatasetEvaluator):
    """
    Evaluate instance calssification results.
    """

    # TODO: unused_arguments: cfg
    def __init__(self, dataset_name, meta, cfg, distributed, output_dir=None, dump=False):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
            meta (SimpleNamespace): dataset metadata.
            cfg (config dict): cvpods Config instance.
            distributed (True): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump all
                results predicted on the dataset. The dump contains:

                1. "instance_predictions.pth" a file in torch serialization
                   format that contains all the raw original predictions.

            dump (bool): If True, after the evaluation is completed, a Markdown file
                that records the model evaluation metrics and corresponding scores
                will be generated in the working directory.
        """
        super(LongTailClassificationEvaluator, self).__init__()
        # TODO: really use dataset_name
        self.dataset_name = dataset_name
        self._dump = dump
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")

        self._metadata = meta
        # for long tail evaluation metrics
        data_root = osp.join(
            osp.split(osp.split(cvpods.__file__)[0])[0], "datasets")
        self._longtail_json = osp.join(data_root, cfg.DATASETS.JSON_PATH)

        self._topk = (1, 5)

    def reset(self):
        self._predictions = []
        self._targets = []
        self._dump_infos = []  # per task

    def process(self, inputs, outputs):
        # Find the top max_k predictions for each sample
        _top_max_k_vals, top_max_k_inds = torch.topk(
            outputs.cpu(), max(self._topk), dim=1, largest=True, sorted=True
        )
        # (batch_size, max_k) -> (max_k, batch_size)
        top_max_k_inds = top_max_k_inds.t()

        self._targets.append(torch.tensor([i["category_id"] for i in inputs]))
        self._predictions.append(top_max_k_inds)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            self._predictions = comm.gather(self._predictions, dst=0)
            self._predictions = list(itertools.chain(*self._predictions))

            self._targets = comm.gather(self._targets, dst=0)
            self._targets = list(itertools.chain(*self._targets))

            if not comm.is_main_process():
                return {}

        if len(self._predictions) == 0:
            logger.warning("[ClassificationEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(self._predictions, f)

        self._results = OrderedDict()
        assert len(self._predictions) == len(self._targets)
        if self._predictions[0] is not None:
            self._eval_classification_accuracy()

        if self._dump:
            _dump_to_markdown(self._dump_infos)

        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def _eval_classification_accuracy(self):
        """
        Evaluate self._predictions on the classification task.
        Fill self._results with the metrics of the tasks.
        """
        batch_size = len(self._targets)

        pred = torch.cat(self._predictions, dim=1)
        target = torch.cat(self._targets)

        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        results = {}
        macro_f1_score = f1_score(
            target.detach().cpu().numpy(),
            pred[0].detach().cpu().numpy(),
            average='macro'
        )
        results["Macro_F1"] = macro_f1_score
        # Update with accuracy of the sub-group
        sub_group_accuracy = self._eval_longtail_subgroup_accuracy(pred, target)
        keys = ['Many', 'Medium', 'Few']

        for iidx, key in enumerate(self._topk):
            correct_k = correct[:key].reshape(-1).float().sum(0, keepdim=True)
            results[f"Top_{key} Acc"] = correct_k.mul_(100.0 / batch_size).item()
            for idx, subgroup in enumerate(keys):
                results[f'Top_{key} {subgroup} Acc'] = sub_group_accuracy[idx][iidx]

        self._results["Accuracy"] = results

        small_table = create_small_table(results)
        logger.info("Evaluation results for classification: \n" + small_table)

        if self._dump:
            dump_info_one_task = {
                "task": "classification",
                "tables": [small_table],
                "dataset": self.dataset_name,
            }
            self._dump_infos.append(dump_info_one_task)

    def _accuracy(self, pred, target):
        batch_size = len(target)
        if batch_size > 0:
            correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        results = []
        for k in self._topk:
            if batch_size > 0:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                results.append(correct_k.mul_(100.0 / batch_size).item())
            else:
                results.append(0.0)

        return results

    def _eval_longtail_subgroup_accuracy(self, preds, target):
        # category_frequency_file = os.path.join(dataset_path,'category_frequency.json')
        with PathManager.open(self._longtail_json, 'r') as f:
            category_frequency = json.load(f)

        many_cats = category_frequency['many_cats']
        medium_cats = category_frequency['medium_cats']
        low_cats = category_frequency['low_cats']

        cat_indicator = torch.zeros(len(self._metadata.thing_classes))
        cat_indicator[many_cats] = 1
        cat_indicator[medium_cats] = 2
        cat_indicator[low_cats] = 3

        labels_group_ids = cat_indicator[target]
        labels_many = target[labels_group_ids == 1]
        labels_medium = target[labels_group_ids == 2]
        labels_low = target[labels_group_ids == 3]

        preds_many = preds[:, labels_group_ids == 1]
        preds_medium = preds[:, labels_group_ids == 2]
        preds_low = preds[:, labels_group_ids == 3]

        many_topks_correct = self._accuracy(preds_many, labels_many)
        medium_topks_correct = self._accuracy(preds_medium, labels_medium)
        low_topks_correct = self._accuracy(preds_low, labels_low)

        top_acc_subgroups = [many_topks_correct, medium_topks_correct, low_topks_correct]

        return top_acc_subgroups


def _dump_to_markdown(dump_infos, md_file="README.md"):
    """
    Dump a Markdown file that records the model evaluation metrics and corresponding scores
    to the current working directory.

    Args:
        dump_infos (list[dict]): dump information for each task.
        md_file (str): markdown file path.
    """
    with open(md_file, "w") as f:
        title = os.path.basename(os.getcwd())
        f.write("# {}  ".format(title))
        for dump_info_per_task in dump_infos:
            task_name = dump_info_per_task["task"]
            dataset_name = dump_info_per_task["dataset"]
            tables = dump_info_per_task["tables"]
            tables = [table.replace("\n", "  \n") for table in tables]
            f.write("\n\n## Evaluation results for {}, Dataset: {}:  \n\n".format(
                task_name, dataset_name)
            )
            f.write(tables[0] + "\n")
