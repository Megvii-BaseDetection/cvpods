#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# This file has been modified by Megvii ("Megvii Modifications").
# All Megvii Modifications are Copyright (C) 2019-2021 Megvii Inc. All rights reserved.
import torch
import torch.nn.functional as F


def aligned_bilinear(tensor: torch.FloatTensor, factor: int):
    """aligned interpolation in feature-level.

    This interpolation method, writen by Adelaidet Condinst Codebase and
        beneficial to seg task, gains 0.5 Mask AP in condinst compared
        with only using F.interpolate(align_corners=True), but it's equivalent
        to only using F.interpolate(align_corners=False) to some extent because
        spatial shape of tensor is even length.
        (https://github.com/aim-uofa/AdelaiDet/blob/
        262010bb87fd40613ed313e1cf48b3dc9211411e/adet/utils/comm.py#L17)
    Adelaidet Codebase Issues url about aligned_bilinear:
        https://github.com/aim-uofa/AdelaiDet/issues/218
    """
    assert factor >= 1
    assert int(factor) == factor
    if factor == 1:
        return tensor
    h, w = tensor.shape[2:]
    # In order to align_corners=True, this method transforms tensor to odd length
    # by padding and applying +1 to target interpolation shape
    tensor = F.pad(tensor, pad=(0, 1, 0, 1), mode="replicate")
    oh = factor * h + 1
    ow = factor * w + 1
    tensor = F.interpolate(
        tensor, size=(oh, ow),
        mode='bilinear',
        align_corners=True
    )

    tensor = F.pad(
        tensor, pad=(factor // 2, 0, factor // 2, 0),
        mode="replicate"
    )
    # clip expected shape
    return tensor[..., :oh - 1, :ow - 1]


def gather_feature(fmap, index, mask=None, use_transform=False):
    """
    used for Centernet
    """
    if use_transform:
        # change a (N, C, H, W) tenor to (N, HxW, C) shape
        batch, channel = fmap.shape[:2]
        fmap = fmap.view(batch, channel, -1).permute((0, 2, 1)).contiguous()

    dim = fmap.size(-1)
    index  = index.unsqueeze(len(index.shape)).expand(*index.shape, dim)
    fmap = fmap.gather(dim=1, index=index)
    if mask is not None:
        # this part is not called in Res18 dcn COCO
        mask = mask.unsqueeze(2).expand_as(fmap)
        fmap = fmap[mask]
        fmap = fmap.reshape(-1, dim)
    return fmap
