#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (C) Megvii, Inc. and its affiliates. All rights reserved.

# This file is modified by the following file:
# https://github.com/facebookresearch/detectron2/blob/main/detectron2/projects/__init__.py
# original file LICENSE: Apache2.0

import importlib
from pathlib import Path

_TOOLS_ROOT = Path(__file__).resolve().parent.parent.parent / "tools"

if _TOOLS_ROOT.is_dir():
    # This is true only for in-place installation (pip install -e, setup.py develop),
    # where setup(package_dir=) does not work: https://github.com/pypa/setuptools/issues/230

    class _ToolsFinder(importlib.abc.MetaPathFinder):

        def find_spec(self, name, path, target=None):
            # pylint: disable=unused-argument
            if not name.startswith("cvpods.tools."):
                return
            project_name = name.split(".")[-1] + ".py"
            target_file = _TOOLS_ROOT / project_name
            if not target_file.is_file():
                return
            return importlib.util.spec_from_file_location(name, target_file)

    import sys
    sys.meta_path.append(_ToolsFinder())
