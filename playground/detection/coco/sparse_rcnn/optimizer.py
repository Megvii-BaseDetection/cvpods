#!/usr/bin/python3
# -*- coding:utf-8 -*-
import itertools

import torch
from torch import optim

from cvpods.solver import OPTIMIZER_BUILDER, OptimizerBuilder


@OPTIMIZER_BUILDER.register()
class FullModelAdamWBuilder(OptimizerBuilder):

    @staticmethod
    def build(model, cfg):
        def maybe_add_full_model_gradient_clipping(optim):
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.FULL_MODEL
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

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

        optimizer = maybe_add_full_model_gradient_clipping(optim.AdamW)(
            param_dicts,
            lr=lr,
            betas=cfg.SOLVER.OPTIMIZER.BETAS,
            weight_decay=cfg.SOLVER.OPTIMIZER.WEIGHT_DECAY,
            amsgrad=cfg.SOLVER.OPTIMIZER.AMSGRAD
        )
        return optimizer
