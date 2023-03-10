#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# This file has been modified by Megvii ("Megvii Modifications").
# All Megvii Modifications are Copyright (C) 2019-2021 Megvii Inc. All rights reserved.

from typing import Any, Dict, List, Set

import torch
from torch import optim

from cvpods.solver import lars_sgd
from cvpods.utils.registry import Registry

OPTIMIZER_BUILDER = Registry("Optimizer builder")

NORM_MODULE_TYPES = (
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.BatchNorm3d,
    torch.nn.SyncBatchNorm,
    # NaiveSyncBatchNorm inherits from BatchNorm2d
    torch.nn.GroupNorm,
    torch.nn.InstanceNorm1d,
    torch.nn.InstanceNorm2d,
    torch.nn.InstanceNorm3d,
    torch.nn.LayerNorm,
    torch.nn.LocalResponseNorm,
)


def exclude_from_wd(named_params, weight_decay, skip_list=['bias', 'bn']):
    params = []
    excluded_params = []
    for name, param in named_params:
        if not param.requires_grad:
            continue
        elif any(layer_name in name for layer_name in skip_list):
            excluded_params.append(param)
        else:
            params.append(param)

    return [
        {'params': params, 'weight_decay': weight_decay},
        {'params': excluded_params, 'weight_decay': 0., 'lars_exclude': True},
    ]


@OPTIMIZER_BUILDER.register()
class OptimizerBuilder:

    @staticmethod
    def build(model, cfg):
        raise NotImplementedError


@OPTIMIZER_BUILDER.register()
class SGDBuilder(OptimizerBuilder):

    @staticmethod
    def build(model, cfg):
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.SOLVER.OPTIMIZER.BASE_LR,
            weight_decay=cfg.SOLVER.OPTIMIZER.WEIGHT_DECAY,
            momentum=cfg.SOLVER.OPTIMIZER.MOMENTUM,
        )
        return optimizer


@OPTIMIZER_BUILDER.register()
class D2SGDBuilder(OptimizerBuilder):

    @staticmethod
    def build(model, cfg):
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module in model.modules():
            for key, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)
                lr = cfg.SOLVER.OPTIMIZER.BASE_LR
                weight_decay = cfg.SOLVER.OPTIMIZER.WEIGHT_DECAY
                if isinstance(module, NORM_MODULE_TYPES):
                    weight_decay = cfg.SOLVER.OPTIMIZER.WEIGHT_DECAY_NORM
                elif key == "bias":
                    # NOTE: unlike Detectron v1, we now default BIAS_LR_FACTOR to 1.0
                    # and WEIGHT_DECAY_BIAS to WEIGHT_DECAY so that bias optimizer
                    # hyperparameters are by default exactly the same as for regular
                    # weights.
                    lr = cfg.SOLVER.OPTIMIZER.BASE_LR * cfg.SOLVER.OPTIMIZER.BIAS_LR_FACTOR
                    weight_decay = cfg.SOLVER.OPTIMIZER.WEIGHT_DECAY
                params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        optimizer = optim.SGD(
            params,
            cfg.SOLVER.OPTIMIZER.BASE_LR,
            momentum=cfg.SOLVER.OPTIMIZER.MOMENTUM
        )
        return optimizer


@OPTIMIZER_BUILDER.register()
class LARS_SGDBuilder(OptimizerBuilder):

    @staticmethod
    def build(model, cfg):
        exclude = cfg.SOLVER.OPTIMIZER.get("WD_EXCLUDE_BN_BIAS", False)
        if exclude:
            param = exclude_from_wd(
                model.named_parameters(), cfg.SOLVER.OPTIMIZER.WEIGHT_DECAY
            )
        else:
            param = model.parameters()
        optimizer = lars_sgd.LARS_SGD(
            param,
            lr=cfg.SOLVER.OPTIMIZER.BASE_LR,
            momentum=cfg.SOLVER.OPTIMIZER.MOMENTUM,
            weight_decay=cfg.SOLVER.OPTIMIZER.WEIGHT_DECAY,
            nesterov=cfg.SOLVER.OPTIMIZER.get("NESTERROV", False),
            eta=cfg.SOLVER.OPTIMIZER.TRUST_COEF,
            eps=cfg.SOLVER.OPTIMIZER.EPS,
        )
        return optimizer


@OPTIMIZER_BUILDER.register()
class AdamBuilder(OptimizerBuilder):

    @staticmethod
    def build(model, cfg):
        lr = cfg.SOLVER.OPTIMIZER.BASE_LR
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=cfg.SOLVER.OPTIMIZER.WEIGHT_DECAY,
            amsgrad=cfg.SOLVER.OPTIMIZER.AMSGRAD
        )
        return optimizer


@OPTIMIZER_BUILDER.register()
class AdamWBuilder(OptimizerBuilder):

    @staticmethod
    def build(model, cfg):
        lr = cfg.SOLVER.OPTIMIZER.BASE_LR
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=cfg.SOLVER.OPTIMIZER.BETAS,
            weight_decay=cfg.SOLVER.OPTIMIZER.WEIGHT_DECAY,
            amsgrad=cfg.SOLVER.OPTIMIZER.AMSGRAD
        )
        return optimizer


@OPTIMIZER_BUILDER.register()
class SGDGateLRBuilder(OptimizerBuilder):
    """
    SGD Gate LR optimizer builder, used for DynamicRouting in cvpods.
    This optimizer will ultiply lr for gating function.
    """

    @staticmethod
    def build(model, cfg):
        gate_lr_multi = cfg.SOLVER.OPTIMIZER.GATE_LR_MULTI
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for name, module in model.named_modules():
            for key, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)
                lr = cfg.SOLVER.OPTIMIZER.BASE_LR
                weight_decay = cfg.SOLVER.OPTIMIZER.WEIGHT_DECAY
                if isinstance(module, NORM_MODULE_TYPES):
                    weight_decay = cfg.SOLVER.OPTIMIZER.WEIGHT_DECAY_NORM
                elif key == "bias":
                    # NOTE: unlike Detectron v1, we now default BIAS_LR_FACTOR to 1.0
                    # and WEIGHT_DECAY_BIAS to WEIGHT_DECAY so that bias optimizer
                    # hyperparameters are by default exactly the same as for regular
                    # weights.
                    lr = cfg.SOLVER.OPTIMIZER.BASE_LR * cfg.SOLVER.OPTIMIZER.BIAS_LR_FACTOR
                    weight_decay = cfg.SOLVER.OPTIMIZER.WEIGHT_DECAY

                if gate_lr_multi > 0.0 and "gate_conv" in name:
                    lr *= gate_lr_multi

                params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        optimizer = torch.optim.SGD(
            params,
            cfg.SOLVER.OPTIMIZER.BASE_LR,
            momentum=cfg.SOLVER.OPTIMIZER.MOMENTUM
        )
        return optimizer


@OPTIMIZER_BUILDER.register()
class AngularSGDBuilder(OptimizerBuilder):

    @staticmethod
    def build(model, cfg):
        match_keys = ["res2", "res3", "res4", "res5"]
        weight_decay = cfg.SOLVER.OPTIMIZER.WEIGHT_DECAY_NORM
        lr = cfg.SOLVER.OPTIMIZER.BASE_LR

        norm_pg = [
            {'params': [], 'name': f'layer{x}', 'weight_decay': weight_decay, "lr": lr}
            for x in range(1, 1 + len(match_keys))
        ]
        other_pg: List[Dict[str, Any]] = []

        def get_match_idx(name):
            for i, key in enumerate(match_keys):
                if key in name:
                    return i

        memo: Set[torch.nn.parameter.Parameter] = set()
        for name, module in model.named_modules():
            for key, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue

                if isinstance(module, NORM_MODULE_TYPES):
                    weight_decay = cfg.SOLVER.OPTIMIZER.WEIGHT_DECAY_NORM
                    lr = cfg.SOLVER.OPTIMIZER.BASE_LR
                    match_idx = get_match_idx(name)
                    if match_idx is None:
                        other_pg += [{
                            "params": [module.weight, module.bias],
                            "lr": lr, "weight_decay": weight_decay
                        }]
                    else:
                        norm_pg[match_idx]["params"].extend([module.weight, module.bias])

                    memo.add(module.weight)
                    memo.add(module.bias)
                    continue
                elif key == "bias":
                    lr = cfg.SOLVER.OPTIMIZER.BASE_LR * cfg.SOLVER.OPTIMIZER.BIAS_LR_FACTOR
                    weight_decay = cfg.SOLVER.OPTIMIZER.WEIGHT_DECAY

                other_pg += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
                memo.add(value)

        optimizer = lars_sgd.AugularSGD(
            norm_pg + other_pg,
            cfg.SOLVER.OPTIMIZER.BASE_LR,
            momentum=cfg.SOLVER.OPTIMIZER.MOMENTUM
        )
        return optimizer
