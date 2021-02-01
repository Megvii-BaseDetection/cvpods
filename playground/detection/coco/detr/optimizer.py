#!/usr/bin/python3
# -*- coding:utf-8 -*-

from torch import optim

from cvpods.solver import OPTIMIZER_BUILDER, OptimizerBuilder


@OPTIMIZER_BUILDER.register()
class DETRAdamWBuilder(OptimizerBuilder):

    @staticmethod
    def build(model, cfg):
        lr = cfg.SOLVER.OPTIMIZER.BASE_LR

        param_dicts = [
            {
                "params": [
                    param for name, param in model.named_parameters()
                    if "backbone" not in name and param.requires_grad]
            },
            {
                "params": [
                    param for name, param in model.named_parameters()
                    if "backbone" in name and param.requires_grad],
                "lr": cfg.SOLVER.OPTIMIZER.BASE_LR_RATIO_BACKBONE * lr,
            },
        ]

        optimizer = optim.AdamW(
            param_dicts,
            lr=lr,
            betas=cfg.SOLVER.OPTIMIZER.BETAS,
            weight_decay=cfg.SOLVER.OPTIMIZER.WEIGHT_DECAY,
            amsgrad=cfg.SOLVER.OPTIMIZER.AMSGRAD
        )
        return optimizer
