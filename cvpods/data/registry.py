#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2019-2021 Megvii Inc. All rights reserved.

from cvpods.utils import Registry

DATASETS = Registry("datasets")
TRANSFORMS = Registry("transforms")
SAMPLERS = Registry("samplers")
PATH_ROUTES = Registry("path_routes")
