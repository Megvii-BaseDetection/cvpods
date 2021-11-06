#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2019-2021 Megvii Inc. All rights reserved.

import math

import cv2
import numpy as np

import torch
import torch.nn.functional as F

from cvpods.modeling.losses import dice_loss
from cvpods.modeling.meta_arch.condinst import CondInst
from cvpods.structures import ImageList
from cvpods.utils import log_first_n


class BoxInst(CondInst):
    """
    Implement BoxInst (https://arxiv.org/abs/2012.02310).
    Below are implementation of boxinst, which is mainly adopted from AdelaiDet.
    https://github.com/aim-uofa/AdelaiDet/tree/master/adet/modeling/condinst
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        # fmt: off
        # BoxInst parameters
        self.enable_color_sup = cfg.MODEL.BOXINST.PAIRWISE_LOSS.ENABLE
        self.patch_size = cfg.MODEL.BOXINST.PAIRWISE_LOSS.PATCH_SIZE
        self.dilation = cfg.MODEL.BOXINST.PAIRWISE_LOSS.DILATION
        self.theta = cfg.MODEL.BOXINST.PAIRWISE_LOSS.THETA
        self.color_sim_thresh = cfg.MODEL.BOXINST.PAIRWISE_LOSS.COLOR_SIM_THRESH
        self.bottom_pixels_removed = cfg.MODEL.BOXINST.BOTTOM_PIXELS_REMOVED
        self.warmup_iter = cfg.MODEL.BOXINST.PAIRWISE_LOSS.WARMUP_ITERS
        self.edge_count_once = cfg.MODEL.BOXINST.PAIRWISE_LOSS.EDGE_COUNT_ONCE
        if self.enable_color_sup:
            self.register_buffer("_iter", torch.zeros([1]))  # warmup iteration
        # fmt: on
        self.to(self.device)

    def forward(self, batched_inputs):
        if self.training and self.enable_color_sup:
            self._iter += 1
            images, lab_images, padded_mask = self.preprocess_training_image(batched_inputs)
        else:
            images = self.preprocess_image(batched_inputs)

        if "instances" in batched_inputs[0]:
            gt_instances = [
                x["instances"].to(self.device) for x in batched_inputs
            ]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                "WARNING",
                "'targets' in the model inputs is now renamed to 'instances'!",
                n=10)
            gt_instances = [
                x["targets"].to(self.device) for x in batched_inputs
            ]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)
        features = [features[f] for f in self.in_features]
        box_cls, box_delta, box_center, box_param = self.head(features)
        shifts = self.shift_generator(features)

        if self.training:
            gt_classes, gt_shifts_reg_deltas, gt_centerness, \
                gt_inst_masks, gt_inds, im_inds, fpn_levels = \
                self.get_ground_truth(
                    shifts, gt_instances, images.tensor
                )
            proposal_losses, proposals = self.proposals_losses(
                gt_classes, gt_shifts_reg_deltas,
                gt_centerness, gt_inds, im_inds,
                box_cls, box_delta, box_center,
                box_param, fpn_levels, shifts
            )
            proposals = self.generate_instance_masks(
                features, shifts, proposals
            )
            projection_losses = self.projection_losses(gt_inst_masks, proposals,
                                                       dummy_feature=box_cls[0])

            losses = {}
            losses.update(proposal_losses)
            losses.update(projection_losses)

            if self.enable_color_sup:
                gt_affinity_mask = self.get_affinity_ground_truth(
                    gt_inst_masks, images.tensor, lab_images, padded_mask)
                affinity_losses = self.affinity_losses(gt_affinity_mask, proposals,
                                                       dummy_feature=box_cls[0])
                losses.update(affinity_losses)

            return losses
        else:
            proposals = self.proposals_inference(
                box_cls, box_delta, box_center,
                box_param, shifts, images
            )
            proposals = self.generate_instance_masks(
                features, shifts, proposals
            )
            padded_im_h, padded_im_w = images.tensor.shape[-2:]
            processed_results = []
            for im_id, (input_per_image, image_size) in \
                    enumerate(zip(batched_inputs, images.image_sizes)):
                im_h = input_per_image.get("height", image_size[0])
                im_w = input_per_image.get("width", image_size[1])
                resized_in_h, resized_in_w = image_size
                instances_per_im = proposals[proposals.im_inds == im_id]
                instances_per_im = self.postprocess(
                    instances_per_im, im_h, im_w,
                    resized_in_h, resized_in_w,
                    padded_im_h, padded_im_w
                )
                processed_results.append({
                    "instances": instances_per_im
                })

            return processed_results

    def preprocess_training_image(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images_norm = [self.normalizer(x) for x in images]
        images_norm = ImageList.from_tensors(images_norm, self.backbone.size_divisibility)
        # get shape infomation
        unpadded_im_shape = [x.shape[1] for x in images]
        ori_im_shape = [x["height"] for x in batched_inputs]
        # build padded_mask for ignore bottom edge and extra padded edge
        padded_mask = [images[0].new_ones(x.shape[1], x.shape[2],
                                          dtype=torch.float) for x in images]
        # padding, downsampling upsampled images and transforming to LAB space
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        downsampled_images = F.avg_pool2d(
            images.tensor.float(), kernel_size=self.mask_out_stride,
            stride=self.mask_out_stride, padding=0
        )
        lab_images = torch.stack([self.bgr_to_lab(x) for x in downsampled_images])

        # Mask out the bottom area where the COCO dataset probably has wrong annotations
        # This trick is adopted by adelaidet's boxinst codebase.
        # In fact, this trick has no influence to final result.
        for i in range(len(padded_mask)):
            pixels_removed = int(
                self.bottom_pixels_removed * unpadded_im_shape[i] / ori_im_shape[i]
            )
            if pixels_removed > 0:
                padded_mask[i][-pixels_removed:, :] = 0
        padded_mask = ImageList.from_tensors(padded_mask, self.backbone.size_divisibility)
        padded_mask = padded_mask.tensor.unsqueeze(1)  # B,H,W -> B,1,H,W
        return images_norm, lab_images, padded_mask

    def bgr_to_lab(self, image):
        # Although adelaidet adopts skimage library to transform image to LAB space,
        # opencv supports faster implementation of RGB2LAB and the average discrepancies
        # of LAB images generated by two method are less than 0.1.
        image = image.permute(1, 2, 0).contiguous().cpu().numpy().astype(np.float32) / 255.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        image = torch.as_tensor(image, device=self.device, dtype=torch.float32).permute(2, 0, 1)
        return image

    @torch.no_grad()
    def get_ground_truth(self, shifts, targets, images):

        gt_classes = []
        gt_shifts_deltas = []
        gt_centerness = []
        gt_inst_masks = []
        gt_inds = []
        fpn_levels = []
        im_inds = []

        num_targets = 0
        im_h, im_w = images.shape[-2:]
        nearest_offset = int(self.mask_out_stride // 2)

        for im_i, (shifts_per_image, targets_per_image) \
                in enumerate(zip(shifts, targets)):
            object_sizes_of_interest = torch.cat([
                shifts_i.new_tensor(size).unsqueeze(0).expand(
                    shifts_i.shape[0], -1) for shifts_i, size in zip(
                    shifts_per_image, self.object_sizes_of_interest)
            ], dim=0)

            shifts_over_all_feature_maps = torch.cat(shifts_per_image, dim=0)

            gt_boxes = targets_per_image.gt_boxes
            deltas = self.shift2box_transform.get_deltas(
                shifts_over_all_feature_maps, gt_boxes.tensor.unsqueeze(1))

            # ground truth for box masks
            gt_instances_masks_i = []
            is_in_boxes = []

            for ind in range(len(gt_boxes)):
                # down-sampling operation and building is_in_boxes per instance to saving memory
                # self.boxes_to_bitmask return shape: 1, im_h, im_w
                bitmask = self.boxes_to_bitmask(gt_boxes[ind].tensor[0],
                                                im_h, im_w).to(self.device)
                # 1, len(shifts)
                is_in_boxes_i = self.generate_in_box_mask(gt_boxes[ind], bitmask,
                                                          deltas[ind: ind + 1],
                                                          im_h, im_w, shifts_per_image)
                # nearest sample to supervised resolution
                bitmask = bitmask[:,
                                  nearest_offset::self.mask_out_stride,
                                  nearest_offset::self.mask_out_stride
                                  ]  # 1, mask_h, mask_w

                is_in_boxes.append(is_in_boxes_i)
                gt_instances_masks_i.append(bitmask)

            is_in_boxes = torch.cat(is_in_boxes, dim=0)  # len(GT), len(shifts)
            gt_instances_masks_i = torch.cat(gt_instances_masks_i, dim=0)  # len(GT), im_h, im_w

            max_deltas = deltas.max(dim=-1).values
            # limit the regression range for each location
            is_cared_in_the_level = \
                (max_deltas >= object_sizes_of_interest[None, :, 0]) & \
                (max_deltas <= object_sizes_of_interest[None, :, 1])

            gt_positions_area = gt_boxes.area().unsqueeze(1).repeat(
                1, shifts_over_all_feature_maps.shape[0]
            )
            gt_positions_area[~is_in_boxes] = math.inf
            gt_positions_area[~is_cared_in_the_level] = math.inf

            # if there are still more than one objects for a position,
            # we choose the one with minimal area
            positions_min_area, gt_matched_idxs = gt_positions_area.min(dim=0)

            gt_ind_i = num_targets + gt_matched_idxs
            num_targets += len(targets_per_image)

            # ground truth box regression
            gt_shifts_reg_deltas_i = self.shift2box_transform.get_deltas(
                shifts_over_all_feature_maps, gt_boxes[gt_matched_idxs].tensor
            )

            # ground truth classes
            has_gt = len(targets_per_image) > 0
            if has_gt:
                gt_classes_i = targets_per_image.gt_classes[gt_matched_idxs]
                # Shifts with area inf are treated as background.
                gt_classes_i[positions_min_area == math.inf] = self.num_classes
            else:
                gt_classes_i = torch.zeros_like(gt_matched_idxs) + self.num_classes

            # ground truth centerness
            left_right = gt_shifts_reg_deltas_i[:, [0, 2]]
            top_bottom = gt_shifts_reg_deltas_i[:, [1, 3]]
            gt_centerness_i = torch.sqrt(
                (left_right.min(dim=-1).values / left_right.max(dim=-1).values).clamp_(min=0)
                * (top_bottom.min(dim=-1).values / top_bottom.max(dim=-1).values).clamp_(min=0)
            )

            fpn_level_i = torch.cat([
                loc.new_ones(len(loc), dtype=torch.long) * level
                for level, loc in enumerate(shifts_per_image)
            ])
            im_ind_i = fpn_level_i.new_ones(len(fpn_level_i)) * im_i

            gt_classes.append(gt_classes_i)
            gt_shifts_deltas.append(gt_shifts_reg_deltas_i)
            gt_centerness.append(gt_centerness_i)
            gt_inst_masks.append(gt_instances_masks_i)
            gt_inds.append(gt_ind_i)
            im_inds.append(im_ind_i),
            fpn_levels.append(fpn_level_i)

        return torch.stack(gt_classes), \
            torch.stack(gt_shifts_deltas), \
            torch.stack(gt_centerness), \
            gt_inst_masks, \
            torch.stack(gt_inds), \
            torch.stack(im_inds), \
            torch.stack(fpn_levels)

    def boxes_to_bitmask(self, gt_box_i, im_h, im_w):
        bitmask = gt_box_i.new_zeros((1, im_h, im_w), dtype=torch.bool)
        bitmask[:, gt_box_i[1].long(): gt_box_i[3].ceil().long(),
                gt_box_i[0].long(): gt_box_i[2].ceil().long()] = True
        return bitmask

    @torch.no_grad()
    def get_affinity_ground_truth(self, gt_inst_masks, images, lab_images, padded_mask):

        gt_inst_affinity_mask = []

        bs, _, im_h, im_w = images.shape
        mask_h, mask_w = im_h // self.mask_out_stride, im_w // self.mask_out_stride
        nearest_offset = int(self.mask_out_stride // 2)
        k = self.patch_size * self.patch_size
        # the aim of padded_mask is to filter extra padded edge when using F.unfold
        padded_mask = padded_mask[...,
                                  nearest_offset::self.mask_out_stride,
                                  nearest_offset::self.mask_out_stride
                                  ]

        padded_mask = F.unfold(padded_mask,  # Float type is required
                               kernel_size=self.patch_size,
                               dilation=self.dilation,
                               padding=(self.dilation * (self.patch_size - 1) // 2)
                               ).reshape(bs, k, mask_h, mask_w)
        padded_mask = torch.cat([padded_mask[:, :k // 2], padded_mask[:, k // 2 + 1:]], dim=1)

        for im_i, (gt_instances_masks_per_image, lab_per_image, padded_mask_per_image) \
                in enumerate(zip(gt_inst_masks, lab_images, padded_mask)):
            # build E_in mask
            center_relation = gt_instances_masks_per_image.unsqueeze(1)  # len(gt), 1, h, w
            padded_mask_per_image = padded_mask_per_image.unsqueeze(0).bool()  # 1,8,h,w
            # the numbers of edges across the boundary will calculate.
            # the two methods differ in the times of involved edges
            # which are across the boundary
            if self.edge_count_once:
                E_in = padded_mask_per_image & center_relation
            else:
                n_insts = len(gt_instances_masks_per_image)
                mask_relation = F.unfold(center_relation,
                                         kernel_size=self.patch_size,
                                         dilation=self.dilation,
                                         padding=(self.dilation * (self.patch_size - 1) // 2)
                                         ).reshape(n_insts, -1, mask_h, mask_w).bool()
                neighbor_relation = torch.cat([mask_relation[:, :k // 2],
                                               mask_relation[:, k // 2 + 1:]],
                                              dim=1).bool()
                E_in = (neighbor_relation | center_relation) * padded_mask_per_image  # bool

            # generate LAB GT
            img_affinity = self.compute_img_affinity_mask(lab_per_image,
                                                          mask_h, mask_w
                                                          ) * E_in

            gt_inst_affinity_mask.append(img_affinity)

        return torch.cat(gt_inst_affinity_mask)

    def compute_img_affinity_mask(self, lab_image, mask_h, mask_w):  # lab_image:  3, H, W

        lab_image = F.unfold(lab_image.unsqueeze(0),  # 1, 3, H, W
                             kernel_size=self.patch_size,
                             dilation=self.dilation,
                             padding=(self.dilation * (self.patch_size - 1) // 2)
                             )
        k = self.patch_size * self.patch_size
        lab_image = lab_image.reshape(1, 3, k, mask_h, mask_w)
        center_inst_mask = lab_image[:, :, k // 2: k // 2 + 1, :]
        neightbor_inst_mask = torch.cat([lab_image[:, :, :k // 2, :],
                                         lab_image[:, :, k // 2 + 1:, :]], dim=2)
        affinity = torch.exp(-torch.norm(center_inst_mask - neightbor_inst_mask,
                                         dim=1) / self.theta)
        affinity_mask = affinity >= self.color_sim_thresh
        return affinity_mask

    def projection_losses(self, gt_masks, proposals, dummy_feature):
        """
        Arguments:
            gt_masks (List[Tensor]):
                a list of N elements, where N = the number of GT instances per batch and
                shape of each elements in gt_masks is (1, mask_h, mask_w)
                segmentation ground truth where the value inside box is 1, outside box is 0
            proposals (Instances):
                A Instances class contains all sampled foreground information per batch,
                thus len(proposals) depends on select_instances function. Two terms are
                required for loss computation when len(proposals) > 0.
                "gt_inds" term, len(proposals) elements, stores mapping relation between
                    predicted instance and gt instance.
                "pred_global_logits" term, shape (len(proposals), 1, mask_h, mask_w),
                    stores predicted logits of foreground segmentation
            dummy_feature (Tensor): a tensor with "requires_grad" equal to True,
                only be used when len(proposals) == 0
        Returns:
            dict[str: Tensor]:
            mapping from a named loss to a scalar tensor
            storing the loss. Used during training only. The dict key is: "loss_proj"
        """
        if len(proposals):
            gt_inds = proposals.gt_inds
            pred_instances_mask = proposals.pred_global_logits.sigmoid()
            # gather gt_masks based on gt_inds
            # gt_masks shape: List(Tensor)
            gt_masks = torch.cat(gt_masks)[gt_inds].to(dtype=pred_instances_mask.dtype)
            gt_proj_x = gt_masks.max(dim=1)[0]
            gt_proj_y = gt_masks.max(dim=2)[0]
            # transform pred mask to compute loss
            # projections pred
            pred_x_proj = pred_instances_mask.squeeze(1).max(dim=1)[0]
            pred_y_proj = pred_instances_mask.squeeze(1).max(dim=2)[0]

            loss_proj = dice_loss(pred_x_proj, gt_proj_x) + \
                dice_loss(pred_y_proj, gt_proj_y)
            loss_proj = loss_proj.mean()
        else:
            loss_proj = dummy_feature.sum() * 0.
        return {
            "loss_proj": loss_proj
        }

    def affinity_losses(self, gt_affinity_masks, proposals, dummy_feature):
        """
        Arguments:
            gt_affinity_masks (Tensor):
                Shape (N, 8, mask_h, mask_w), where N = the number of GT instances per batch
                color affinity ground truth
            proposals (Instances):
                A Instances class contains all sampled foreground information per batch,
                thus len(proposals) depends on select_instances function. Two terms are
                required for loss computation when len(proposals) > 0.
                "gt_inds" term, len(proposals) elements, stores mapping relation between
                    predicted instance and gt instance.
                "pred_global_logits" term, shape (len(proposals), 1, mask_h, mask_w),
                    stores predicted logits of foreground segmentation
            dummy_feature (Tensor): a tensor with "requires_grad" equal to True,
                only be used when len(proposals) == 0
        Returns:
            dict[str: Tensor]:
            mapping from a named loss to a scalar tensor
            storing the loss. Used during training only. The dict key is: "loss_affinity"
        """
        if len(proposals):
            gt_inds = proposals.gt_inds
            pred_instances_logits = proposals.pred_global_logits
            # gather gt based on gt_inds
            gt_affinity_masks = gt_affinity_masks[gt_inds]
            affinity_pred = self.get_affinity_pred(pred_instances_logits)

            loss_affinity = (
                affinity_pred * gt_affinity_masks
            ).sum() / max(gt_affinity_masks.sum(), 1.0)
            # warmup_iter trick avoids trivial solution (masks of all the pixels being 0 or 1).
            warmup_factor = min(self._iter.item() / self.warmup_iter, 1.0)
            loss_affinity = loss_affinity * warmup_factor
        else:
            loss_affinity = dummy_feature.sum() * 0.
        return {
            "loss_affinity": loss_affinity
        }

    def get_affinity_pred(self, mask_logits):
        N, _, H, W = mask_logits.shape
        k = self.patch_size * self.patch_size
        # transform multiplication in sigmoid space to add operation in log sigmoid space
        log_fg_prob = F.logsigmoid(mask_logits)
        log_bg_prob = F.logsigmoid(-mask_logits)
        log_fg_prob_unfold = F.unfold(log_fg_prob,
                                      kernel_size=self.patch_size,
                                      dilation=self.dilation,
                                      padding=(self.dilation * (self.patch_size - 1) // 2)
                                      ).reshape(N, k, H, W)
        log_bg_prob_unfold = F.unfold(log_bg_prob,
                                      kernel_size=self.patch_size,
                                      dilation=self.dilation,
                                      padding=(self.dilation * (self.patch_size - 1) // 2)
                                      ).reshape(N, k, H, W)
        log_fg_neighbor = torch.cat([log_fg_prob_unfold[:, :k // 2],
                                     log_fg_prob_unfold[:, k // 2 + 1:]], dim=1)
        log_bg_neighbor = torch.cat([log_bg_prob_unfold[:, :k // 2],
                                     log_bg_prob_unfold[:, k // 2 + 1:]], dim=1)
        log_same_fg_prob = log_fg_prob + log_fg_neighbor
        log_same_bg_prob = log_bg_prob + log_bg_neighbor
        # ensure numerical instability of log operation
        max_ = torch.max(log_same_fg_prob, log_same_bg_prob)
        log_same_prob = torch.log(
            torch.exp(log_same_fg_prob - max_) + torch.exp(log_same_bg_prob - max_)
        ) + max_
        return -log_same_prob
