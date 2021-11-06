#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2019-2021 Megvii Inc. All rights reserved.

from torch.utils.data import Sampler

from ..registry import SAMPLERS


@SAMPLERS.register()
class Infinite(Sampler):
    r"""
    Infinite Sampler wraper for basic sampler. Code of pytorch dataloader will be autoloaded after
    sampler raise a StopIteration, we will check code of torch in the futures. Use Infinite wrapper
    to work around temporarily.
    """

    def __init__(self, sampler):
        self.epoch = 0
        self.sampler = sampler
        self.sampler_iter = iter(self.sampler)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            index = next(self.sampler_iter)
        except StopIteration:
            self.epoch += 1
            if callable(getattr(self.sampler, "set_epoch", None)):
                # use set_epoch to prevent generating the same sampler index
                self.sampler.set_epoch(self.epoch)
            self.sampler_iter = iter(self.sampler)
            index = next(self.sampler_iter)
        return index
