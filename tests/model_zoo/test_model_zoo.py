# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import unittest

from cvpods import model_zoo
from cvpods.modeling import FPN, GeneralizedRCNN

logger = logging.getLogger(__name__)


class TestModelZoo(unittest.TestCase):
    def test_get_returns_model(self):
        model = model_zoo.get(
            "examples/segmentation/coco/rcnn/mask_rcnn.res50.fpn.coco.multiscale.1x", trained=False
        )
        self.assertIsInstance(model, GeneralizedRCNN)
        self.assertIsInstance(model.backbone, FPN)

    def test_get_invalid_model(self):
        self.assertRaises(RuntimeError, model_zoo.get, "Invalid/config.yaml")

    def test_get_url(self):
        url = model_zoo.get_checkpoint_s3uri(
            "examples/detection/coco/rcnn/faster_rcnn.res50.fpn.coco.multiscale.1x"
        )
        self.assertEqual(
            url,
            "s3://generalDetection/cvpods/model_zoo/examples/detection/coco/rcnn/faster_rcnn.res50.fpn.coco.multiscale.1x/model_final.pth",  # noqa
        )


if __name__ == "__main__":
    unittest.main()
