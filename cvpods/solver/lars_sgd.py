#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2019-2021 Megvii Inc. All rights reserved.

import torch
from torch.optim.optimizer import Optimizer, required


class LARS_SGD(Optimizer):
    r"""Implements layer-wise adaptive rate scaling for SGD.
    Based on Algorithm 1 of the following paper by You, Gitman, and Ginsburg.
    Large Batch Training of Convolutional Networks:
        https://arxiv.org/abs/1708.03888
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float): base learning rate (\gamma_0)
        momentum (float, optional): momentum factor (default: 0) ("m")
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        eta (float, optional): LARS coefficient
    Example:
        >>> optimizer = LARS(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4, eta=1e-3)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """

    def __init__(self, params, lr=required, momentum=0.9, dampening=0,
                 weight_decay=1e-4, eta=1e-3, eps=1e-8, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if eta < 0.0:
            raise ValueError("Invalid LARS coefficient value: {}".format(eta))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, eta=eta, eps=eps, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        super(LARS_SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(LARS_SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]
            lr = group["lr"]

            eta = group.get("eta", 1e-3)
            eps = group.get("eps", 1e-8)
            lars_exclude = group.get("lars_exclude", False)

            for p in group["params"]:
                if p.grad is None:
                    continue

                d_p = p.grad.data

                if not lars_exclude:
                    weight_norm = torch.norm(p.data)
                    grad_norm = torch.norm(d_p)
                    if weight_norm != 0 and grad_norm != 0:
                        local_lr = eta * weight_norm / (
                            grad_norm + weight_decay * weight_norm + eps)
                    else:
                        local_lr = 1.0
                    actual_lr = local_lr * lr
                else:
                    actual_lr = lr

                d_p = d_p.add(p, alpha=weight_decay).mul(actual_lr)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf
                p.add_(-d_p)

        return loss
