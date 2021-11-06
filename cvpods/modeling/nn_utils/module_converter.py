#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2019-2021 Megvii Inc. All rights reserved.

from loguru import logger

from torch import nn

from cvpods.layers import FrozenBatchNorm2d, batch_norm
from cvpods.utils.distributed import get_world_size

SYNC_BN_MODULE = (
    nn.SyncBatchNorm,
    batch_norm.NaiveSyncBatchNorm,
    batch_norm.NaiveSyncBatchNorm1d,
)


def maybe_convert_module(model):
    if get_world_size() == 1:
        logger.warning("SyncBN used with 1GPU, auto convert to BatchNorm")
        model = convert_syncbn(model)

    return model


def convert_syncbn(module):
    """convert SyncBatchNorm in a module to BatchNorm

    Args:
        module (nn.Module): module to convert SyncBatchNorm to BatchNorm.

    Return:
        model (nn.Module): converted module.
    """
    model = module

    if isinstance(module, SYNC_BN_MODULE):
        if isinstance(module, batch_norm.NaiveSyncBatchNorm1d):
            model = nn.BatchNorm1d(module.num_features)
        else:
            model = nn.BatchNorm2d(module.num_features)

        # copy data value
        if module.affine:
            model.weight.data = module.weight.data.clone().detach()
            model.bias.data = module.bias.data.clone().detach()

        model.running_mean.data = module.running_mean.data
        model.running_var.data = module.running_var.data
        model.eps = module.eps
        # copy train/eval status
        model.training = module.training
        for name, param in module.named_parameters():
            model_param = getattr(model, name)
            model_param.requires_grad = param.requires_grad

    else:  # convert syncbn to bn recurrisvely
        for name, child in module.named_children():
            new_child = convert_syncbn(child)
            if new_child is not child:
                model.add_module(name, new_child)

    return model


def freeze_module_until(module, name):
    """
    Freeze all submodules inplace before and including `name`. If `isintance(name, list)`,
    submodules before and including every `name` will be frozen.

    Args:
        module (nn.Module): torch.nn.Module
        name (list(str)): submodule name
    """
    def _is_children(name1, name2):
        return name1.startswith(name2) and name1 != name2

    def _freeze_module(module):
        for p in module.parameters():
            p.requires_grad = False
        return FrozenBatchNorm2d.convert_frozen_batchnorm(module)

    if isinstance(name, str):
        name = [name]

    model = module

    for module_name, module in model.named_modules():
        if module_name == "" or any([_is_children(n, module_name) for n in name]):
            continue
        module = _freeze_module(module)

        # direct child
        if len(module_name.split(".")) == 1:
            model.add_module(module_name, module)

        if module_name in name:
            return

    raise ValueError(
        "The `{}` module is not found.".format(name)
    )
