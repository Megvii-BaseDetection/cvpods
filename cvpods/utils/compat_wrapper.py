#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2019-2021 Megvii Inc. All rights reserved.
import functools
from types import FunctionType
from loguru import logger


def deprecated_func(extra_info=None):

    def deprecated(func):
        @functools.wraps(func)
        def wrap_deprecated(*args, **kwargs):
            logger.warning("{} will be deprecated. {}".format(func.__name__, extra_info))
            res = func(*args, **kwargs)
            return res

        return wrap_deprecated

    return deprecated


def deprecated_cls(extra_info=None):

    def deprecated(cls):
        def __init__(self, *args, **kwargs):
            logger.warning("{} will be deprecated. {}".format(cls.__name__, extra_info))
            super(cls, self).__init__(*args, **kwargs)

        cls.__init__ = __init__

        return cls

    return deprecated


def deprecated(extra_info=None):

    def deprecated(input):
        if isinstance(input, FunctionType):
            return deprecated_func(extra_info)(input)
        else:
            return deprecated_cls(extra_info)(input)

    return deprecated
