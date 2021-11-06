#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2019-2021 Megvii Inc. All rights reserved.

from collections import OrderedDict

import timm
from torch import nn

from cvpods.layers import ShapeSpec
from cvpods.modeling.backbone import Backbone
from cvpods.modeling.nn_utils import freeze_module_until


class TIMMBackbone(Backbone):
    def __init__(
        self,
        name,
        pretrained,
        input_channels,
        num_classes=None,
        extra_head=None,
        out_features=None,
    ):
        super().__init__()
        m = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)

        if "vit" not in name:
            out_indices = [i for i, dct in enumerate(m.feature_info) if dct["module"] != ""]

            self.feature_extractor = timm.create_model(
                name,
                pretrained=pretrained,
                in_chans=input_channels,
                features_only=True,
                out_indices=out_indices
            )

            self._out_feature_channels = OrderedDict()
            self._out_feature_strides = OrderedDict()
            for info_dict in self.feature_extractor.feature_info.get_dicts():
                self._out_feature_channels[info_dict["module"]] = info_dict["num_chs"]
                self._out_feature_strides[info_dict["module"]] = info_dict["reduction"]

            final_stage = info_dict["module"]
        else:
            self.feature_extractor = timm.create_model(
                name,
                pretrained=pretrained,
                in_chans=input_channels,
                num_classes=0
            )
            self._out_feature_channels = OrderedDict({
                "pre_logits": self.feature_extractor.num_features
            })
            self._out_feature_strides = OrderedDict({"pre_logits": None})
            final_stage = "pre_logits"

        self.num_classes = num_classes
        if self.num_classes is not None:
            self.extra_head = extra_head
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) if "vit" not in name else nn.Identity()
            self.linear = m.get_classifier()
            final_stage = "linear"

        self._out_features = [final_stage] if out_features is None else out_features

        valid_features = list(self._out_feature_channels.keys())
        valid_features += ["linear"] if self.num_classes is not None else []

        for name in self._out_features:
            assert name in valid_features, \
                "Output feature `{}` not founded among {}".format(name, valid_features)

    def forward(self, x):
        outputs = {}
        features = self.feature_extractor(x)

        if not isinstance(features, list):
            features = [features]

        module_names = self._out_feature_channels.keys()
        for name, feature in zip(module_names, features):
            if name in self._out_features:
                outputs[name] = feature

        if self.num_classes is not None:
            x = features[-1]
            if self.extra_head is not None:
                x = self.extra_head(x)
            x = self.avgpool(x)

            if isinstance(self.linear, nn.Linear):
                x = self.linear(x.flatten(1))
            else:
                x = self.linear(x).flatten(1)

            if "linear" in self._out_features:
                outputs["linear"] = x

        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            ) if name != 'linear' else
            ShapeSpec(channels=self.num_classes)
            for name in self._out_features
        }

    def freeze(self, freeze_at):
        if freeze_at < 1:
            return

        freeze_at_stage = list(self._out_feature_channels.keys())[freeze_at - 1]
        freeze_at_stage_renamed = freeze_at_stage.replace(".", "_")

        if isinstance(self.feature_extractor, timm.models.features.FeatureHookNet) \
           and list(self.feature_extractor.keys()) == ["body"]:
            body = self.feature_extractor.body
        else:
            body = self.feature_extractor

        freeze_module_until(body, [freeze_at_stage, freeze_at_stage_renamed])


def _build_extra_module(name, pretrained):
    if "mobilenetv2" in name \
       or "mnasnet" in name \
       or "fbnetc" in name \
       or "spnasnet" in name \
       or "efficientnet" in name \
       or "mixnet" in name:
        m = timm.create_model(name, pretrained=pretrained)
        extra = nn.Sequential(
            m.conv_head,
            m.bn2,
            m.act2
        )
        return extra

    if "hrnet" in name:
        extra = timm.create_model(name, pretrained=pretrained).final_layer
        return extra

    if "mobilenetv3" in name:
        m = timm.create_model(name, pretrained=pretrained)
        extra = nn.Sequential(
            m.global_pool,
            m.conv_head,
            m.act2
        )
        return extra

    if "rexnet" in name:
        m = timm.create_model(name, pretrained=pretrained)
        return m.features[-1]

    if "vgg" in name and "repvgg" not in name:
        return timm.create_model(name, pretrained=pretrained).pre_logits

    return None


def build_timm_backbone(cfg, input_shape):
    if "prune" in cfg.MODEL.TIMM.NAME:
        raise ValueError("Pruned model is not supported")
    if "vit" in cfg.MODEL.TIMM.NAME and "distilled" in cfg.MODEL.TIMM.NAME:
        raise ValueError("Distilled VIT is not supported")

    extra = _build_extra_module(cfg.MODEL.TIMM.NAME, pretrained=cfg.MODEL.TIMM.PRETRAINED)

    model = TIMMBackbone(
        name=cfg.MODEL.TIMM.NAME,
        pretrained=cfg.MODEL.TIMM.PRETRAINED,
        input_channels=input_shape.channels,
        num_classes=cfg.MODEL.TIMM.NUM_CLASSES,
        extra_head=extra,
        out_features=cfg.MODEL.TIMM.OUT_FEATURES
    )

    model.freeze(cfg.MODEL.BACKBONE.FREEZE_AT)

    return model
