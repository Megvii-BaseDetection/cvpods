# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import contextlib
import io
import numpy as np
import os
import unittest
from collections import defaultdict
import torch
import tqdm
from cvpods.utils import benchmark
from pycocotools.coco import COCO
from tabulate import tabulate
from torch.nn import functional as F

from cvpods.layers.mask_ops import (
    pad_masks,
    paste_mask_in_image_old,
    paste_masks_in_image,
    scale_boxes,
)
from cvpods.structures import BitMasks, Boxes, BoxMode, PolygonMasks
from cvpods.structures.masks import polygons_to_bitmask


def iou_between_full_image_bit_masks(a, b, eps=0.001):
    intersect = (a & b).sum() + eps
    union = (a | b).sum() + eps
    return intersect / union


def rasterize_polygons_with_grid_sample(full_image_bit_mask, box, mask_size, threshold=0.5):
    x0, y0, x1, y1 = box[0], box[1], box[2], box[3]

    img_h, img_w = full_image_bit_mask.shape

    mask_y = np.arange(0.0, mask_size) + 0.5  # mask y sample coords in [0.5, mask_size - 0.5]
    mask_x = np.arange(0.0, mask_size) + 0.5  # mask x sample coords in [0.5, mask_size - 0.5]
    mask_y = (mask_y) / (mask_size) * (y1 - y0) + y0
    mask_x = (mask_x) / (mask_size) * (x1 - x0) + x0

    mask_x = (mask_x - 0.5) / (img_w - 1) * 2 + -1
    mask_y = (mask_y - 0.5) / (img_h - 1) * 2 + -1
    gy, gx = torch.meshgrid(torch.from_numpy(mask_y), torch.from_numpy(mask_x))
    ind = torch.stack([gx, gy], dim=-1).to(dtype=torch.float32)

    full_image_bit_mask = torch.from_numpy(full_image_bit_mask)
    mask = F.grid_sample(
        full_image_bit_mask[None, None, :, :].to(dtype=torch.float32),
        ind[None, :, :, :],
        align_corners=True,
    )

    return mask[0, 0] >= threshold


class TestMaskCropPaste(unittest.TestCase):
    def setUp(self):
        json_file = "datasets/coco/annotations/instances_train2017.json"
        if not os.path.isfile(json_file):
            print("{} not found".format(json_file))
            raise unittest.SkipTest("{} not found".format(json_file))
        with contextlib.redirect_stdout(io.StringIO()):
            self.coco = COCO(json_file)

    def test_crop_paste_consistency(self):
        """
        rasterize_polygons_within_box (used in training)
        and
        paste_masks_in_image (used in inference)
        should be inverse operations to each other.

        This function runs several implementation of the above two operations and prints
        the reconstruction error.
        """

        anns = self.coco.loadAnns(self.coco.getAnnIds(iscrowd=False))  # avoid crowd annotations

        selected_anns = anns[:100]

        ious = []
        for ann in tqdm.tqdm(selected_anns):
            results = self.process_annotation(ann)
            ious.append([k[2] for k in results])

        ious = np.array(ious)
        mean_ious = ious.mean(axis=0)
        table = []
        res_dic = defaultdict(dict)
        for row, iou in zip(results, mean_ious):
            table.append((row[0], row[1], iou))
            res_dic[row[0]][row[1]] = iou
        print(tabulate(table, headers=["rasterize", "paste", "iou"], tablefmt="simple"))
        # assert that the reconstruction is good:
        self.assertTrue(
            res_dic["polygon"]["aligned"] > 0.94,
            msg="res_dic['polygon']['aligned']: {}".format(res_dic["polygon"]["aligned"])
        )
        self.assertTrue(
            res_dic["roialign"]["aligned"] > 0.95,
            msg="res_dic['roialign']['aligned']: {}".format(res_dic["roialign"]["aligned"])
        )

    def process_annotation(self, ann, mask_side_len=28):
        # Parse annotation data
        img_info = self.coco.loadImgs(ids=[ann["image_id"]])[0]
        height, width = img_info["height"], img_info["width"]
        gt_polygons = [np.array(p, dtype=np.float64) for p in ann["segmentation"]]
        gt_bbox = BoxMode.convert(ann["bbox"], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
        gt_bbox = np.array(gt_bbox)
        gt_bit_mask = polygons_to_bitmask(gt_polygons, height, width)

        # Run rasterize ..
        torch_gt_bbox = torch.Tensor(gt_bbox)[None, :].to(dtype=torch.float32)
        box_bitmasks = {
            "polygon": PolygonMasks([gt_polygons]).crop_and_resize(torch_gt_bbox, mask_side_len)[0],
            "gridsample": rasterize_polygons_with_grid_sample(gt_bit_mask, gt_bbox, mask_side_len),
            "roialign": BitMasks(torch.from_numpy(gt_bit_mask[None, :, :])).crop_and_resize(
                torch_gt_bbox, mask_side_len
            )[0],
        }

        # Run paste ..
        results = defaultdict(dict)
        for k, box_bitmask in box_bitmasks.items():
            padded_bitmask, scale = pad_masks(box_bitmask[None, :, :], 1)
            scaled_boxes = scale_boxes(torch_gt_bbox, scale)

            r = results[k]
            r["old"] = paste_mask_in_image_old(
                padded_bitmask[0], scaled_boxes[0], height, width, threshold=0.5
            )
            r["aligned"] = paste_masks_in_image(
                box_bitmask[None, :, :], Boxes(gt_bbox[None, :]), (height, width)
            )[0]

        table = []
        for rasterize_method, r in results.items():
            for paste_method, mask in r.items():
                mask = np.asarray(mask)
                iou = iou_between_full_image_bit_masks(gt_bit_mask.astype("uint8"), mask)
                table.append((rasterize_method, paste_method, iou))
        return table

    def test_polygon_area(self):
        # Draw polygon boxes
        for d in [5.0, 10.0, 1000.0]:
            polygon = PolygonMasks([[[0, 0, 0, d, d, d, d, 0]]])
            area = polygon.area()[0]
            target = d ** 2
            self.assertEqual(area, target)

        # Draw polygon triangles
        for d in [5.0, 10.0, 1000.0]:
            polygon = PolygonMasks([[[0, 0, 0, d, d, d]]])
            area = polygon.area()[0]
            target = d ** 2 / 2
            self.assertEqual(area, target)


def benchmark_paste():
    S = 800
    H, W = image_shape = (S, S)
    N = 64
    torch.manual_seed(42)
    masks = torch.rand(N, 28, 28)

    center = torch.rand(N, 2) * 600 + 100
    wh = torch.clamp(torch.randn(N, 2) * 40 + 200, min=50)
    x0y0 = torch.clamp(center - wh * 0.5, min=0.0)
    x1y1 = torch.clamp(center + wh * 0.5, max=S)
    boxes = Boxes(torch.cat([x0y0, x1y1], axis=1))

    def func(device, n=3):
        m = masks.to(device=device)
        b = boxes.to(device=device)

        def bench():
            for _ in range(n):
                paste_masks_in_image(m, b, image_shape)
            if device.type == "cuda":
                torch.cuda.synchronize()

        return bench

    specs = [{"device": torch.device("cpu"), "n": 3}]
    if torch.cuda.is_available():
        specs.append({"device": torch.device("cuda"), "n": 3})

    benchmark(func, "paste_masks", specs, num_iters=10, warmup_iters=2)


if __name__ == "__main__":
    benchmark_paste()
    unittest.main()
