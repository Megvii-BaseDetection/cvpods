#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2019-2021 Megvii Inc. All rights reserved.

import functools
import itertools
from collections import OrderedDict, defaultdict, namedtuple
from enum import IntEnum, unique
import tabulate

import numpy as np

import torch
import torch.autograd.profiler as tprofiler

# profile info in table
COMMON_STAT = [
    "self_cpu_time", "cpu_time", "self_cuda_time", "cuda_time",
    "self_cpu_mem", "cpu_mem", "self_cuda_mem", "cuda_mem", "hits"
]
STAT_TYPE = ["float32"] * (len(COMMON_STAT) - 1) + ["int32"]  # "hits" is int32 type
LEADING_KEY = "  "   # leading string of module name LAYER_TREE profile mode

LeafModule = namedtuple("LeafModule", ["path", "module"])
ProfileInfo = namedtuple("ProfileInfo", COMMON_STAT)


@unique
class ProfileMode(IntEnum):

    def __new__(cls, value, doc=None):
        self = int.__new__(cls, value)
        self._value_ = value
        if doc is not None:
            self.__doc__ = doc
        return self

    LAYER = 1, "Layer by layer profile"
    OP = 2, "Operator level profile"
    MIXED = 3, "Operator of layer level profile"
    LAYER_TREE = 4, "Layer level profile, presented in tree format"


def leaf_modules_generator(module, name=None, path=None):
    """
    Generate all leaf modules of an given pytorch module

    Args:
        module (nn.Module): a pytorch nn.Module
        name (string): name of pytorch module
        path (Tuple[string]): path to pytorch module

    Return:
        Generator contains LeafModule.
    """
    if path is None:
        path = ()

    if not name:
        name = module.__class__.__name__
    named_children = list(module.named_children())
    path = path + (name,)
    if len(named_children) == 0:
        yield LeafModule(".".join(path), module)
    # recursively walk into all submodules
    for name, child_module in named_children:
        yield from leaf_modules_generator(child_module, name=name, path=path)


class profile:
    """
    Profile tool for Pytorch models, using torch.autograd.profile inside.
    """

    def __init__(
        self,
        module,
        enabled=True,
        use_cuda=False,
        paths=None,
        profile_memory=False,
        mode=ProfileMode.LAYER
    ):
        """
        Args:
            module (nn.Module): torch module to profile.
            enabled (bool): whether enable profile or not.
            use_cuda (bool): whether use cuda profiler or not.
            paths (Iterable[string]): profile paths for hooks. For example, if you want see
                profile info of modeule named classifier, use path=("classifier")
            profile_memory (bool): whether profile memory or not, require torch >= 1.6
            mode (IntEnum): ProfileMode enum.
        """
        self._module = module
        self.enabled = enabled
        self.use_cuda = use_cuda
        self.paths = paths

        if profile_memory:
            torch_ver = [int(x) for x in torch.__version__.split(".", maxsplit=2)[:2]]
            assert torch_ver >= [1, 6], "profile_memory = True requires torch 1.6+"
        self.profile_memory = profile_memory
        assert mode in ProfileMode, "Profile mode {} not found".format(mode)
        self.mode = mode

        self.entered = False
        self.exited = False
        self.traces = ()
        self.profile_events = defaultdict(list)

    def __enter__(self):
        if not self.enabled:
            return self
        if self.entered:
            raise RuntimeError("profile is not reentrant")
        self.entered = True
        self._forwards = {}  # store the original forward functions
        self.traces = tuple(map(self._add_profile_hook, leaf_modules_generator(self._module)))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled:
            return
        tuple(map(self._remove_profile_hook, self.traces))
        del self._forwards  # remove unnecessary forwards
        self.exited = True

    def _traces_to_table(self, **kwargs):
        """
        human readable output of the profiler traces and events.
        """
        tree_dict = build_info_tree(self.traces, self.profile_events, self.mode)

        if self.mode == ProfileMode.LAYER_TREE:
            max_depth = kwargs.pop("max_depth", 3)
            tree_dict = format_to_module_tree(tree_dict, max_depth)
            headers, data = generate_header_and_data(tree_dict)
            table = TreeTable(headers, data, max_depth=max_depth)
        else:
            headers, data = generate_header_and_data(tree_dict)
            table = InfoTable(headers, data)

        return table

    def table(
        self, sorted_by="cpu_time", row_limit=None, average=False, with_percent=True, **kwargs
    ):
        """
        return profile info in table format

        Args:
            sorted_by (string): which data the table is sorted by. Default sorted by cpu time
            row_limit (int): row limit number of table, None means no limit.
            average (bool): whether average profile data by hits or not.
            with_percent (bool): whether profile data presented with percent data or not.
            kwargs:
                max_depth (int): depth of tree if using LAYER_TREE mode
        """
        if not self.exited:
            return "<unfinished profile>"
        else:
            table = self._traces_to_table(**kwargs)
            if not self.use_cuda:
                table.filter([x for x in COMMON_STAT if "cuda" in x])

            if self.mode != ProfileMode.LAYER_TREE:
                # sorted/row limit/avarage not supported for TreeInfo type
                table = table.sorted_by(sorted_by).row_limit(row_limit)

                if average:
                    table.average()

            table.with_percent = with_percent
            return table

    def _add_profile_hook(self, leaf_module):

        def cond(path, paths):
            return sum([key in path for key in paths])

        path, module = leaf_module
        if (self.paths is not None and cond(path, self.paths)) or (self.paths is None):
            _forward = module.forward
            self._forwards[path] = _forward

            @functools.wraps(_forward)
            def wrap_forward(*args, **kwargs):
                with tprofiler.profile(
                    use_cuda=self.use_cuda, profile_memory=self.profile_memory,
                ) as prof:
                    res = _forward(*args, **kwargs)
                event_list = prof.function_events
                event_list.populate_cpu_children()
                # each profile call should be contained in its own list
                self.profile_events[path].append(event_list)
                return res

            module.forward = wrap_forward

        return leaf_module

    def _remove_profile_hook(self, trace):
        path, module = trace
        if (self.paths is not None and path in self.paths) or (self.paths is None):
            module.forward = self._forwards[path]


class InfoTable:

    def __init__(self, headers, data, with_percent=False):
        """
        Args:
            header (Iterable[string]): header of info table.
            data (Iterable[numpy.array]): data of table.
            with_percent (bool): whether data presented with percent data or not.
        """
        assert len(headers) == len(data), "length of headers and data are not matched"
        self.headers = headers
        self.info = {key: value for key, value in zip(headers, data)}
        self.with_percent = with_percent

    def insert(self, header, data, position=-1):
        """
        insert header and data into current table at given position

        Args:
            header (Iterable[string]): inserted data.
            data (Iterable[numpy.array]): inserted data.
            position (int): insert position, the same usage like list indexing.
        """

        def swap(a, b):
            a, b = b, a

        self.info[header] = data
        if header in self.headers:
            index = self.headers.index(header)
            swap(self.headers[index], self.headers[position])
        else:
            self.headers.insert(position, header)

    def sorted_by(self, keyname=None, descending=True):
        """
        use keyname to sort table.

        Args:
            keyname (string): sorted header name.
            descending (bool): whether sorted in descending order or not.
        """
        if keyname is None:
            return self
        if keyname not in self.info:
            keyname += "_avg"
        assert keyname in self.info
        sort_index = np.argsort(self.info[keyname], axis=0).reshape(-1)
        if descending:
            sort_index = sort_index[::-1]
        for header in self.headers:
            self.info[header] = self.info[header][sort_index]

        return self

    def filter(self, filter_list=None):
        """
        filter header and data in filter list.

        Args:
            filter_list (Iterable[string]): list of headers that needs to be filtered out
        """
        self.headers = [header for header in self.headers if header not in filter_list]

    def filter_zeros(self):
        """filter all zeros data."""
        filter_list = []
        for header in self.headers:
            data = self.info[header]
            if "float" in data.dtype.name or "int" in data.dtype.name:
                if data.sum() == 0:
                    filter_list.append(header)
        self.filter(filter_list)

    def average(self, average_key="hits"):
        """
        average table with average key

        Args:
            average_key:
        """
        hits = self.info[average_key]
        for i, header in enumerate(self.headers):
            if header.endswith("time") or header.endswith("mem"):
                self.info[header + "_avg"] = self.info[header] / hits
                self.headers[i] += "_avg"
                del self.info[header]

    def row_limit(self, limit=None):
        """
        return table in row limit format.

        Args:
            limit (int): row limit number of table, None means no limit.
        """
        if limit is None:
            return self
        else:
            data = [self.info[x][:limit] for x in self.headers]
            return InfoTable(headers=self.headers, data=data)

    def __repr__(self):
        self.filter_zeros()
        time_formatter = np.vectorize(tprofiler.format_time)
        mem_formatter = np.vectorize(tprofiler.format_memory)
        percent_formatter = np.vectorize(lambda x: " ({:.2%})".format(x))

        fmt_data = []
        for header in self.headers:
            data = self.info[header]
            if "time" in header:
                time_array = time_formatter(data)
                if self.with_percent:
                    percent = percent_formatter(data / data.sum())
                    time_array = np.core.defchararray.add(time_array, percent)
                fmt_data.append(time_array)
            elif "mem" in header:
                mem_array = mem_formatter(data)
                if self.with_percent:
                    percent = percent_formatter(data / data.sum())
                    mem_array = np.core.defchararray.add(mem_array, percent)
                fmt_data.append(mem_array)
            else:
                fmt_data.append(data)

        concat_data = np.concatenate(fmt_data, axis=1)
        table = tabulate.tabulate(concat_data, headers=self.headers, tablefmt="fancy_grid")
        return table


class TreeTable(InfoTable):

    def __init__(self, headers, data, with_percent=False, max_depth=3):
        super().__init__(headers, data, with_percent)
        self.max_depth = max_depth

    def __repr__(self):
        self.filter_zeros()
        time_formatter = np.vectorize(tprofiler.format_time)
        mem_formatter = np.vectorize(tprofiler.format_memory)
        percent_formatter = np.vectorize(lambda x: " ({:.2%})".format(x))

        fmt_data = []
        for header in self.headers:
            data = self.info[header]
            if "time" in header:
                time_array = time_formatter(data)
                if self.with_percent:
                    percent = percent_formatter(data / data[0])  # data[0] is the sum value
                    time_array = np.core.defchararray.add(time_array, percent)
                fmt_data.append(time_array)
            elif "mem" in header:
                mem_array = mem_formatter(data)
                if self.with_percent:
                    percent = percent_formatter(data / data[0])  # sum value, ditto
                    mem_array = np.core.defchararray.add(mem_array, percent)
                fmt_data.append(mem_array)
            else:
                fmt_data.append(data)

        concat_data = np.concatenate(fmt_data, axis=1)

        # white space should be kept under profile
        old_ws = tabulate.PRESERVE_WHITESPACE
        tabulate.PRESERVE_WHITESPACE = True
        table = tabulate.tabulate(concat_data, headers=self.headers, tablefmt="fancy_grid")
        tabulate.PRESERVE_WHITESPACE = old_ws
        return table


def generate_header_and_data(tree_dict):
    headers = ["name"] + COMMON_STAT

    format_lines = [
        (
            name,
            info.self_cpu_time,
            info.cpu_time,
            info.self_cuda_time,
            info.cuda_time,
            info.self_cpu_mem,
            info.cpu_mem,
            info.self_cuda_mem,
            info.cuda_mem,
            info.hits,
        ) for name, info in tree_dict.items()
    ]
    data = np.array(format_lines)
    data = np.hsplit(data, len(headers))
    data[1:] = [x.astype(dtype) for x, dtype in zip(data[1:], STAT_TYPE)]
    return headers, data


def format_to_module_tree(profile_dict, max_depth=3):

    def merge_info(origin_info, update_info):
        sum_result = tuple(a + b for a, b in zip(origin_info[:-1], update_info[:-1]))
        hits = (max(origin_info[-1], update_info[-1]), )
        return sum_result + hits

    tree_format_dict = OrderedDict()
    for key, info in profile_dict.items():
        path = [k for i, k in enumerate(key.split(".", maxsplit=max_depth))]
        path = list(itertools.accumulate(path, lambda x, y: x + "." + y))[:max_depth]
        path_with_whitespace = [LEADING_KEY * i + k for i, k in enumerate(path)]

        for p in path_with_whitespace:
            if p in tree_format_dict:
                tree_format_dict[p] = ProfileInfo(*merge_info(info, tree_format_dict[p]))
            else:
                tree_format_dict[p] = info

    return tree_format_dict


def get_profile_info(events, path_events):
    info = ProfileInfo(
        # TIME
        sum([e.self_cpu_time_total for e in events]),
        sum([e.cpu_time_total for e in events]),
        sum([e.self_cuda_time_total for e in events]),
        sum([e.cuda_time_total for e in events]),
        # Memory
        sum([e.self_cpu_memory_usage for e in events]),
        sum([e.cpu_memory_usage for e in events]),
        sum([e.self_cuda_memory_usage for e in events]),
        sum([e.cuda_memory_usage for e in events]),
        # Hits
        len(path_events)
    )
    return info


def build_info_tree(traces, trace_events, mode=ProfileMode.LAYER):
    """
    build profile dict according to profile mode.
    """
    assert mode in ProfileMode, "ProfileMode {} not found".format(mode)
    tree = OrderedDict()

    for trace in traces:
        path, module = trace
        # unwrap all of the events, in case model is called multiple times
        events = [te for tevents in trace_events[path] for te in tevents]
        if mode == ProfileMode.LAYER or mode == ProfileMode.LAYER_TREE:
            tree[path] = get_profile_info(events, trace_events[path])
        elif mode == ProfileMode.OP or mode == ProfileMode.MIXED:
            for op in set(event.name for event in events):
                op_events = [e for e in events if e.name == op]
                stat = get_profile_info(op_events, op_events)
                if mode == ProfileMode.MIXED:
                    tree[path + "." + op] = stat
                else:  # operator mode
                    if op not in tree:  # init op in tree
                        tree[op] = stat
                    else:
                        # add Op level profile info to original value.
                        tree[op] = ProfileInfo(*(a + b for a, b in zip(tree[op], stat)))

    return tree
