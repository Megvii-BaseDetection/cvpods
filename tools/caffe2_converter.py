#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# This file has been modified by Megvii ("Megvii Modifications").
# All Megvii Modifications are Copyright (C) 2019-2021 Megvii Inc. All rights reserved.
import argparse
import os

from cvpods.checkpoint import DefaultCheckpointer
from cvpods.config import get_cfg
from cvpods.data import build_test_loader
from cvpods.evaluation import COCOEvaluator, inference_on_dataset, print_csv_format
from cvpods.export import add_export_config, export_caffe2_model
from cvpods.modeling import build_model
from cvpods.utils import setup_logger


def setup_cfg(args):
    cfg = get_cfg()
    # cuda context is initialized before creating dataloader, so we don't fork anymore
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg = add_export_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a model to Caffe2")
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--run-eval", action="store_true")
    parser.add_argument("--output", help="output directory for the converted caffe2 model")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    logger = setup_logger()
    logger.info("Command line arguments: " + str(args))

    cfg = setup_cfg(args)

    # create a torch model
    torch_model = build_model(cfg)
    DefaultCheckpointer(torch_model).resume_or_load(cfg.MODEL.WEIGHTS)

    # get a sample data
    data_loader = build_test_loader(cfg, cfg.DATASETS.TEST[0])
    first_batch = next(iter(data_loader))

    # convert and save caffe2 model
    caffe2_model = export_caffe2_model(cfg, torch_model, first_batch)
    caffe2_model.save_protobuf(args.output)
    # draw the caffe2 graph
    caffe2_model.save_graph(os.path.join(args.output, "model.svg"), inputs=first_batch)

    # run evaluation with the converted model
    if args.run_eval:
        dataset = cfg.DATASETS.TEST[0]
        data_loader = build_test_loader(cfg, dataset)
        # NOTE: hard-coded evaluator. change to the evaluator for your dataset
        evaluator = COCOEvaluator(dataset, cfg, True, args.output)
        metrics = inference_on_dataset(caffe2_model, data_loader, evaluator)
        print_csv_format(metrics)
