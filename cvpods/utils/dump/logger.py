#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# This file has been modified by Megvii ("Megvii Modifications").
# All Megvii Modifications are Copyright (C) 2019-2021 Megvii Inc. All rights reserved.
import datetime
import itertools
import os
import sys
import time
from collections import Counter
from loguru import logger
from tabulate import tabulate

from cvpods.utils import ensure_dir


def str_timestamp(time_value=None):
    """format given timestamp, if no timestamp is given, return a call time string"""
    if time_value is None:
        time_value = datetime.datetime.now()
    return time_value.strftime("%Y-%m-%d_%H-%M-%S")


def setup_logger(output=None, distributed_rank=0):
    """
    Initialize the cvpods logger and set its verbosity level to "INFO".

    Args:
        output (str): a file name or a directory to save log. If None, will not save log file.
            If ends with ".txt" or ".log", assumed to be a file name.
            Otherwise, logs will be saved to `output/log.txt`.

    Returns:
        logging.Logger: a logger
    """
    logger.remove()
    loguru_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )

    # stdout logging: master only
    if distributed_rank == 0:
        logger.add(sys.stderr, format=loguru_format)

    # file logging: all workers
    if output is not None:
        if output.endswith(".txt") or output.endswith(".log"):
            filename = output
        else:
            filename = os.path.join(output, "log.txt")
        if distributed_rank > 0:
            filename = filename + ".rank{}".format(distributed_rank)
        ensure_dir(os.path.dirname(filename))
        logger.add(filename)


"""
Below are some other convenient logging methods.
They are mainly adopted from
https://github.com/abseil/abseil-py/blob/master/absl/logging/__init__.py
"""


def _find_caller():
    """
    Returns:
        str: module name of the caller
        tuple: a hashable key to be used to identify different callers
    """
    frame = sys._getframe(2)
    while frame:
        code = frame.f_code
        if os.path.join("utils", "writer", "logger.") not in code.co_filename:
            mod_name = frame.f_globals["__name__"]
            if mod_name == "__main__":
                mod_name = "cvpods"
            return mod_name, (code.co_filename, frame.f_lineno, code.co_name)
        frame = frame.f_back


_LOG_COUNTER = Counter()
_LOG_TIMER = {}


def log_first_n(lvl, msg, n=1, *, key="caller"):
    """
    Log only for the first n times.

    Args:
        lvl (int): the logging level
        msg (str):
        n (int):
        key (str or tuple[str]): the string(s) can be one of "caller" or
            "message", which defines how to identify duplicated logs.
            For example, if called with `n=1, key="caller"`, this function
            will only log the first call from the same caller, regardless of
            the message content.
            If called with `n=1, key="message"`, this function will log the
            same content only once, even if they are called from different places.
            If called with `n=1, key=("caller", "message")`, this function
            will not log only if the same caller has logged the same message before.
    """
    if isinstance(key, str):
        key = (key,)
    assert len(key) > 0

    caller_module, caller_key = _find_caller()
    hash_key = ()
    if "caller" in key:
        hash_key = hash_key + caller_key
    if "message" in key:
        hash_key = hash_key + (msg,)

    _LOG_COUNTER[hash_key] += 1
    if _LOG_COUNTER[hash_key] <= n:
        logger.opt(depth=1).log(lvl, msg)


def log_every_n(lvl, msg, n=1):
    """
    Log once per n times.

    Args:
        lvl (int): the logging level
        msg (str):
        n (int):
    """
    caller_module, key = _find_caller()
    _LOG_COUNTER[key] += 1
    if n == 1 or _LOG_COUNTER[key] % n == 1:
        logger.opt(depth=1).log(lvl, msg)


def log_every_n_seconds(lvl, msg, n=1):
    """
    Log no more than once per n seconds.
    Args:
        lvl (int): the logging level
        msg (str):
        n (int):
    """
    caller_module, key = _find_caller()
    last_logged = _LOG_TIMER.get(key, None)
    current_time = time.time()
    if last_logged is None or current_time - last_logged >= n:
        logger.opt(depth=1).log(lvl, msg)
        _LOG_TIMER[key] = current_time


def create_small_table(small_dict):
    """
    Create a small table using the keys of small_dict as headers. This is only
    suitable for small dictionaries.

    Args:
        small_dict (dict): a result dictionary of only a few items.

    Returns:
        str: the table as a string.
    """
    keys, values = tuple(zip(*small_dict.items()))
    table = tabulate(
        [values],
        headers=keys,
        tablefmt="pipe",
        floatfmt=".3f",
        stralign="center",
        numalign="center",
    )
    return table


def create_table_with_header(header_dict, headers=["category", "AP"], min_cols=6):
    """
    create a table with given header.

    Args:
        header_dict (dict):
        headers (list):
        min_cols (int):

    Returns:
        str: the table as a string
    """
    assert min_cols % len(headers) == 0, "bad table format"
    num_cols = min(min_cols, len(header_dict) * len(headers))
    result_pair = [x for pair in header_dict.items() for x in pair]
    row_pair = itertools.zip_longest(*[result_pair[i::num_cols] for i in range(num_cols)])
    table = tabulate(
        row_pair, tablefmt="pipe", floatfmt=".3f",
        headers=headers * (num_cols // len(headers)),
        numalign="left")
    return table
