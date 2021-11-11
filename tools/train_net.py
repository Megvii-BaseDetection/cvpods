#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# This file has been modified by Megvii ("Megvii Modifications").
# All Megvii Modifications are Copyright (C) 2019-2021 Megvii Inc. All rights reserved.
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in cvpods.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use cvpods as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""
import os
import pickle as pkl
import sys
from collections import OrderedDict
from loguru import logger

import torch

from cvpods.engine import RUNNERS, default_argument_parser, default_setup, hooks, launch
from cvpods.evaluation import build_evaluator
from cvpods.modeling import GeneralizedRCNNWithTTA
from cvpods.utils import comm


def runner_decrator(cls):
    """
    We use the "DefaultRunner" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleRunner", or write your own training loop. You can use
    "tools/plain_train_net.py" as an example.
    """

    def custom_build_evaluator(cls, cfg, dataset_name, dataset, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        dump_train = cfg.GLOBAL.DUMP_TRAIN
        return build_evaluator(cfg, dataset_name, dataset, output_folder, dump=dump_train)

    def custom_test_with_TTA(cls, cfg, model):
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        res = cls.test(cfg, model, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA"))
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

    cls.build_evaluator = classmethod(custom_build_evaluator)
    cls.test_with_TTA = classmethod(custom_test_with_TTA)

    return cls


def train_argument_parser():
    parser = default_argument_parser()
    parser.add_argument(
        "--dir", type=str, default=None,
        help="path of dir that contains config and network, default to working dir"
    )
    parser.add_argument("--clearml", action="store_true", help="use clearml or not")
    return parser


@logger.catch
def main(args, config, build_model):
    config.merge_from_list(args.opts)
    cfg = default_setup(config, args)

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop or subclassing the runner.
    """
    runner = runner_decrator(RUNNERS.get(cfg.TRAINER.NAME))(cfg, build_model)
    runner.resume_or_load(resume=args.resume)

    extra_hooks = []
    if args.clearml:
        from cvpods.engine.clearml import ClearMLHook
        if comm.is_main_process():
            extra_hooks.append(ClearMLHook())
    if cfg.TEST.AUG.ENABLED:
        extra_hooks.append(
            hooks.EvalHook(0, lambda: runner.test_with_TTA(cfg, runner.model))
        )
    if extra_hooks:
        runner.register_hooks(extra_hooks)

    logger.info("Running with full config:\n{}".format(cfg))
    base_config = cfg.__class__.__base__()
    logger.info("different config with base class:\n{}".format(cfg.diff(base_config)))

    runner.train()

    if comm.is_main_process() and cfg.MODEL.AS_PRETRAIN:
        # convert last ckpt to pretrain format
        convert_to_pretrained_model(
            input=os.path.join(cfg.OUTPUT_DIR, "model_final.pth"),
            save_path=os.path.join(cfg.OUTPUT_DIR, "model_final_pretrain_weight.pkl")
        )


def convert_to_pretrained_model(input, save_path):
    obj = torch.load(input, map_location="cpu")
    obj = obj["model"]

    newmodel = {}
    for k, v in obj.items():
        if not k.startswith("encoder_q.") and not k.startswith("network"):
            continue
        old_k = k
        if k.startswith("encoder_q."):
            k = k.replace("encoder_q.", "")
        elif k.startswith("network"):
            k = k.replace("network.", "")
        print(old_k, "->", k)
        newmodel[k] = v.numpy()

    res = {
        "model": newmodel,
        "__author__": "MOCO" if k.startswith("encoder_q.") else "CLS",
        "matching_heuristics": True
    }

    with open(save_path, "wb") as f:
        pkl.dump(res, f)


if __name__ == "__main__":
    args = train_argument_parser().parse_args()
    if args.num_gpus is None:
        args.num_gpus = torch.cuda.device_count()

    extra_sys_path = ".." if args.dir is None else args.dir
    sys.path.append(extra_sys_path)

    from config import config
    from net import build_model

    config.link_log()
    logger.info("Create soft link to {}".format(config.OUTPUT_DIR))

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args, config, build_model),
    )
