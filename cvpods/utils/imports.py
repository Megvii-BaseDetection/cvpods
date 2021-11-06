#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2019-2021 Megvii Inc. All rights reserved.

import importlib
import os.path as osp


def dynamic_import(config_name, config_path):
    """
    Dynamic import a project.

    Args:
        config_name (str): module name
        config_path (str): the dir that contains the .py with this module.

    Examples::
        >>> root = "/path/to/your/retinanet/"
        >>> project = root + "retinanet.res50.fpn.coco.800size.1x.mrcnn_sigmoid"
        >>> cfg = dynamic_import("config", project).config
        >>> net = dynamic_import("net", project)
    """
    spec = importlib.util.spec_from_file_location("", osp.join(config_path, config_name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

    # return importlib.import_module(config_name, config_path)
