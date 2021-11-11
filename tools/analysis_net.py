#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2019-2021 Megvii Inc. All rights reserved.

import argparse
import os
import sys
from tabulate import tabulate

import torch

from cvpods.analyser.analysis import (
    activation_count_operators,
    flop_count_operators,
    parameter_count_table
)
from cvpods.utils import dynamic_import, setup_logger


def build_network_and_cfg(args, logger):
    cfg_path = args.config_file
    assert cfg_path.endswith(".py")
    path = os.path.dirname(cfg_path)
    sys.path.append(path)
    file_name = os.path.basename(cfg_path).rstrip(".py")
    cfg = dynamic_import(file_name, path).config
    cfg.merge_from_list(args.opts)

    net_file = args.net_file
    assert net_file.endswith(".py")
    path = os.path.dirname(net_file)
    sys.path.append(path)
    file_name = os.path.basename(net_file).rstrip(".py")
    build_func = dynamic_import(file_name, path).build_model
    try:
        model = build_func(cfg)
    except Exception:
        logger.warn("change MODEL.DEVICE from {} to cpu".format(cfg.MODEL.DEVICE))
        cfg.MODEL.DEVICE = "cpu"
        model = build_func(cfg)

    return model, cfg


def arg_parser():
    parser = argparse.ArgumentParser(
        description="A script that analyse provided model"
    )
    parser.add_argument(
        "--config-file", default="./config.py", metavar="FILE", help="path to config file"
    )
    parser.add_argument(
        "--net-file", default="./net.py", metavar="FILE", help="path to network file"
    )
    parser.add_argument("--height", help="Height of input image.", type=int, default=None)
    parser.add_argument("--width", help="Width of input image.", type=int, default=None)
    parser.add_argument("--channel", help="Width of input image.", type=int, default=3)
    parser.add_argument(
        "--depth", help="Max depth of model parameters.", type=int, default=3,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def analyze_model(model, args=None):
    tables_dict = {}
    tables_dict["params table"] = parameter_count_table(model, args.depth)
    if args.height is not None and args.width is not None:
        data = [{"image": torch.randn(args.channel, args.height, args.width)}]
        flops_dict = flop_count_operators(model, data)
        flops_dict["total"] = sum(flops_dict.values())
        tables_dict["flops table"] = tabulate(
            flops_dict.items(), headers=["name", "GFlops"], tablefmt="pipe"
        )
        activation_dict = activation_count_operators(model, data)
        activation_dict["total"] = sum(activation_dict.values())
        tables_dict["activation table"] = tabulate(
            activation_dict.items(), headers=["name", "Mega Act Count"], tablefmt="pipe"
        )
    return tables_dict


def main():
    args = arg_parser().parse_args()
    logger = setup_logger()

    model, _ = build_network_and_cfg(args, logger)
    tables = analyze_model(model, args)
    for k, tab in tables.items():
        logger.info("{}:\n{}\n".format(k, tab))


if __name__ == "__main__":
    main()
