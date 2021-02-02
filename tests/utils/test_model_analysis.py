# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.


import unittest
import torch

import cvpods.model_zoo as model_zoo
# TODO: add cvpods.utils.analysis
# from cvpods.utils.analysis import flop_count_operators, parameter_count


# flake8: noqa
@unittest.skip("cvpods.utils.analysis not implemented")
class RetinaNetTest(unittest.TestCase):
    def setUp(self):
        self.model = model_zoo.get(
            "examples/detection/coco/retinanet/retinanet.res50.fpn.coco.multiscale.1x/"
        )
        print(type(self.model))
    def test_flop(self):
        # RetinaNet supports flop-counting with random inputs
        inputs = [{"image": torch.rand(3, 800, 800)}]
        res = flop_count_operators(self.model, inputs)
        self.assertTrue(int(res["conv"]), 146)  # 146B flops

    def test_param_count(self):
        res = parameter_count(self.model)
        self.assertTrue(res[""], 37915572)
        self.assertTrue(res["backbone"], 31452352)


@unittest.skip("cvpods.utils.analysis not implemented")
class FasterRCNNTest(unittest.TestCase):
    def setUp(self):
        self.model = model_zoo.get(
            "examples/detection/coco/rcnn/faster_rcnn.res50.fpn.coco.multiscale.1x/"
        )
    def test_flop(self):
        # Faster R-CNN supports flop-counting with random inputs
        inputs = [{"image": torch.rand(3, 800, 800)}]
        res = flop_count_operators(self.model, inputs)

        # This only checks flops for backbone & proposal generator
        # Flops for box head is not conv, and depends on #proposals, which is
        # almost 0 for random inputs.
        self.assertTrue(int(res["conv"]), 117)

    def test_param_count(self):
        res = parameter_count(self.model)
        self.assertTrue(res[""], 41699936)
        self.assertTrue(res["backbone"], 26799296)
