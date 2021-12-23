#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# This file has been modified by Megvii ("Megvii Modifications").
# All Megvii Modifications are Copyright (C) 2019-2021 Megvii Inc. All rights reserved.
"""
Testing Script for cvpods.

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
import glob
import os
import re
import sys
from collections import OrderedDict
from pprint import pformat
import megfile
from loguru import logger
import torch

from cvpods.checkpoint import DefaultCheckpointer
from cvpods.engine import RUNNERS, default_argument_parser, default_setup, launch
from cvpods.evaluation import build_evaluator, verify_results
from cvpods.modeling import GeneralizedRCNN, GeneralizedRCNNWithTTA, TTAWarper
from cvpods.utils import comm


def runner_decrator(cls):
    """
    We use the "DefaultRunner" which contains a number pre-defined logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleRunner", or write your own training loop.
    """

    def custom_build_evaluator(cls, cfg, dataset_name, dataset, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        dump_test = cfg.GLOBAL.DUMP_TEST
        return build_evaluator(cfg, dataset_name, dataset, output_folder, dump=dump_test)

    def custom_test_with_TTA(cls, cfg, model):
        # In the end of training, run an evaluation with TTA
        logger.info("Running inference with test-time augmentation ...")

        module = model

        if isinstance(module, comm.DDP_TYPES):
            module = model.module
        if isinstance(module, GeneralizedRCNN):
            model = GeneralizedRCNNWithTTA(cfg, model)
        else:
            model = TTAWarper(cfg, model)
        res = cls.test(cfg, model, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA"))
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

    cls.build_evaluator = classmethod(custom_build_evaluator)
    cls.test_with_TTA = classmethod(custom_test_with_TTA)

    return cls


def test_argument_parser():
    parser = default_argument_parser()
    parser.add_argument(
        "--dir", type=str, default=None,
        help="path of dir that contains config and network, default to working dir"
    )
    parser.add_argument("--start-iter", type=int, default=None, help="start iter used to test")
    parser.add_argument("--end-iter", type=int, default=None, help="end iter used to test")
    parser.add_argument("--debug", action="store_true", help="use debug mode or not")
    return parser


def filter_by_iters(file_list, start_iter, end_iter):
    # sort file_list by modified time
    if file_list[0].startswith("s3://"):
        file_list.sort(key=lambda x: megfile.smart_stat(x).extra["LastModified"])
    else:
        file_list.sort(key=os.path.getmtime)

    if start_iter is None:
        if end_iter is None:
            # use latest ckpt if start_iter and end_iter are not given
            return [file_list[-1]]
        else:
            start_iter = 0
    elif end_iter is None:
        end_iter = float("inf")

    iter_infos = [re.split(r"model_|\.pth", f)[-2] for f in file_list]
    keep_list = [0] * len(iter_infos)
    start_index = 0
    if "final" in iter_infos and iter_infos[-1] != "final":
        start_index = iter_infos.index("final")

    for i in range(len(iter_infos) - 1, start_index, -1):
        if iter_infos[i] == "final":
            if end_iter == float("inf"):
                keep_list[i] = 1
        elif float(start_iter) < float(iter_infos[i]) < float(end_iter):
            keep_list[i] = 1
            if float(iter_infos[i - 1]) > float(iter_infos[i]):
                break

    return [filename for keep, filename in zip(keep_list, file_list) if keep == 1]


def get_valid_files(args, cfg, logger):

    if "MODEL.WEIGHTS" in args.opts:
        model_weights = cfg.MODEL.WEIGHTS
        assert megfile.smart_exists(model_weights), "{} not exist!!!".format(model_weights)
        return [model_weights]

    file_list = glob.glob(os.path.join(cfg.OUTPUT_DIR, "model_*.pth"))
    assert len(file_list) > 0, "Plz provide model to evaluate"

    file_list = filter_by_iters(file_list, args.start_iter, args.end_iter)
    assert file_list, "No checkpoint valid in {}.".format(cfg.OUTPUT_DIR)
    logger.info("All files below will be tested in order:\n{}".format(pformat(file_list)))
    return file_list


@logger.catch
def main(args, config, build_model):
    config.merge_from_list(args.opts)
    cfg = default_setup(config, args)
    if args.num_gpus is None:
        args.num_gpus = torch.cuda.device_count()
    if args.debug:
        batches = int(cfg.SOLVER.IMS_PER_DEVICE * args.num_gpus)
        if cfg.SOLVER.IMS_PER_BATCH != batches:
            cfg.SOLVER.IMS_PER_BATCH = batches
            logger.warning("SOLVER.IMS_PER_BATCH is changed to {}".format(batches))

    valid_files = get_valid_files(args, cfg, logger)
    # * means all if need specific format then *.csv
    for current_file in valid_files:
        cfg.MODEL.WEIGHTS = current_file
        model = build_model(cfg)

        DefaultCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        if cfg.TEST.AUG.ENABLED:
            res = runner_decrator(RUNNERS.get(cfg.TRAINER.NAME)).test_with_TTA(cfg, model)
        else:
            res = runner_decrator(RUNNERS.get(cfg.TRAINER.NAME)).test(cfg, model)

        if comm.is_main_process():
            verify_results(cfg, res)

    # return res


if __name__ == "__main__":
    args = test_argument_parser().parse_args()
    logger.info("Command Line Args: {}".format(args))
    if args.num_gpus is None:
        args.num_gpus = torch.cuda.device_count()

    extra_sys_path = ".." if args.dir is None else args.dir
    sys.path.append(extra_sys_path)

    from config import config  # isort:skip  # noqa: E402
    from net import build_model  # isort:skip  # noqa: E402

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args, config, build_model),
    )
