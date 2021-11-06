#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2019-2021 Megvii Inc. All rights reserved.

import math
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import nn

from cvpods.layers import ShapeSpec, cat, generalized_batched_nms, get_norm
from cvpods.modeling.box_regression import Shift2BoxTransform
from cvpods.modeling.losses import dice_loss, iou_loss, sigmoid_focal_loss_jit
from cvpods.modeling.meta_arch.fcos import Scale
from cvpods.modeling.meta_arch.retinanet import permute_to_N_HWA_K
from cvpods.modeling.nn_utils.feature_utils import aligned_bilinear
from cvpods.structures import Boxes, ImageList, Instances, polygons_to_bitmask
from cvpods.utils import comm, log_first_n


def permute_all_to_N_HWA_K_and_concat(
        box_cls, box_delta, box_center, box_parmas, param_count, num_classes=80,
):
    box_cls_flattened = [permute_to_N_HWA_K(x, num_classes) for x in box_cls]
    box_delta_flattened = [permute_to_N_HWA_K(x, 4) for x in box_delta]
    box_center_flattened = [permute_to_N_HWA_K(x, 1) for x in box_center]
    box_parmas_flattened = [permute_to_N_HWA_K(x, param_count) for x in box_parmas]

    box_cls = cat(box_cls_flattened, dim=1).reshape(-1, num_classes)
    box_delta = cat(box_delta_flattened, dim=1).reshape(-1, 4)
    box_center = cat(box_center_flattened, dim=1).reshape(-1, 1)
    box_parmas = cat(box_parmas_flattened, dim=1).reshape(-1, param_count)
    return box_cls, box_delta, box_center, box_parmas


class CondInst(nn.Module):
    """
    Implement CondInst (https://arxiv.org/abs/2003.05664).
    Below are implementation of condinst, which is mainly adopted from AdelaiDet.
    https://github.com/aim-uofa/AdelaiDet/tree/master/adet/modeling/condinst
    """
    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)
        # fmt: off
        self.num_classes = cfg.MODEL.FCOS.NUM_CLASSES
        self.in_features = cfg.MODEL.FCOS.IN_FEATURES
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.iou_smooth = cfg.MODEL.FCOS.IOU_SMOOTH
        # Condinst parameters
        self.mask_out_stride = cfg.MODEL.CONDINST.MASK_OUT_STRIDE
        self.max_proposals = cfg.MODEL.CONDINST.MAX_PROPOSALS
        self.topk_proposals_per_im = cfg.MODEL.CONDINST.TOPK_PROPOSALS_PER_IM
        assert (self.max_proposals != -1) ^ (self.topk_proposals_per_im != -1),\
            "MAX_PROPOSALS and TOPK_PROPOSALS_PER_IM " \
            "cannot be set to -1 or enabled at the same time."
        self.disable_rel_coords = cfg.MODEL.CONDINST.MASK_HEAD.DISABLE_REL_COORDS
        self.mask_center_sample = cfg.MODEL.CONDINST.MASK_CENTER_SAMPLE
        # Loss parameters:
        self.focal_loss_alpha = cfg.MODEL.FCOS.FOCAL_LOSS_ALPHA
        self.focal_loss_gamma = cfg.MODEL.FCOS.FOCAL_LOSS_GAMMA
        self.iou_loss_type = cfg.MODEL.FCOS.IOU_LOSS_TYPE
        self.center_sampling_radius = cfg.MODEL.FCOS.CENTER_SAMPLING_RADIUS
        # Inference parameters:
        self.thresh_with_centerness = cfg.MODEL.FCOS.THRESH_WITH_CENTERNESS
        self.score_threshold = cfg.MODEL.FCOS.SCORE_THRESH_TEST
        self.topk_candidates = cfg.MODEL.FCOS.TOPK_CANDIDATES_TEST
        self.nms_threshold = cfg.MODEL.FCOS.NMS_THRESH_TEST
        self.nms_type = cfg.MODEL.NMS_TYPE
        self.max_detections_per_image = cfg.TEST.DETECTIONS_PER_IMAGE
        self.infer_mask_threshold = cfg.MODEL.CONDINST.INFER_MASK_THRESH
        # fmt: on

        self.backbone = cfg.build_backbone(
            cfg, input_shape=ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN)))

        backbone_shape = self.backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in self.in_features]

        self.dynamic_mask_head = DynamicMaskHead(cfg)
        self.num_gen_params = self.dynamic_mask_head.num_gen_params

        self.mask_branch = MaskBranch(cfg, backbone_shape)
        self.mask_branch_out_stride = self.mask_branch.out_stride
        self.mask_out_level_ind = self.fpn_strides.index(self.mask_branch_out_stride)

        self.head = CondInstHead(cfg, self.num_gen_params, feature_shapes)
        self.shift_generator = cfg.build_shift_generator(cfg, feature_shapes)

        # Matching and loss
        self.shift2box_transform = Shift2BoxTransform(
            weights=cfg.MODEL.FCOS.BBOX_REG_WEIGHTS)
        self.object_sizes_of_interest = cfg.MODEL.FCOS.OBJECT_SIZES_OF_INTEREST

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).reshape(
            3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).reshape(
            3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def forward(self, batched_inputs):

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
                gt_instance_masks, gt_inds, im_inds, fpn_levels = \
                self.get_ground_truth(
                    shifts, gt_instances, images.tensor.shape[-2:]
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
            instances_losses = self.instances_losses(
                gt_instance_masks, proposals, dummy_feature=box_cls[0])

            losses = {}
            losses.update(proposal_losses)
            losses.update(instances_losses)

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

    def generate_instance_masks(self, features, shifts, proposals):
        """
        Generate Mask Logits by Mask Branch and Dynamic Mask Head.
        """
        mask_input_feats = self.mask_branch(features)

        # get mask shift depend on downsample ratio
        mask_shift = shifts[0][self.mask_out_level_ind]
        proposals = self.dynamic_mask_head(
            mask_input_feats, self.mask_branch_out_stride,
            mask_shift, proposals
        )
        return proposals

    @torch.no_grad()
    def get_ground_truth(self, shifts, targets, image_shape):

        gt_classes = []
        gt_shifts_deltas = []
        gt_centerness = []
        gt_instances_masks = []
        gt_inds = []
        fpn_levels = []
        im_inds = []
        num_targets = 0
        im_h, im_w = image_shape[-2:]
        nearest_offset = int(self.mask_out_stride // 2)
        for im_i, (shifts_per_image, targets_per_image) in enumerate(zip(shifts, targets)):
            object_sizes_of_interest = torch.cat([
                shifts_i.new_tensor(size).unsqueeze(0).expand(
                    shifts_i.shape[0], -1) for shifts_i, size in zip(
                    shifts_per_image, self.object_sizes_of_interest)
            ], dim=0)

            shifts_over_all_feature_maps = torch.cat(shifts_per_image, dim=0)

            gt_boxes = targets_per_image.gt_boxes

            deltas = self.shift2box_transform.get_deltas(
                shifts_over_all_feature_maps, gt_boxes.tensor.unsqueeze(1))

            # ground truth for instances masks
            polygons = targets_per_image.get("gt_masks").polygons
            gt_instances_masks_i = []
            is_in_boxes = []
            for ind in range(len(polygons)):
                # down-sampling operation and building is_in_boxes per instance to saving memory
                bitmask = polygons_to_bitmask(polygons[ind], im_h, im_w)
                bitmask = torch.from_numpy(bitmask).to(self.device).unsqueeze(0)
                # 1, len(shifts)
                is_in_boxes_i = self.generate_in_box_mask(gt_boxes[ind], bitmask,
                                                          deltas[ind: ind + 1],
                                                          im_h, im_w,
                                                          shifts_per_image)
                # nearest sample to supervised resolution
                bitmask = bitmask[:, nearest_offset::self.mask_out_stride,
                                  nearest_offset::self.mask_out_stride]

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
            gt_instances_masks.append(gt_instances_masks_i)
            gt_inds.append(gt_ind_i)
            fpn_levels.append(fpn_level_i)
            im_inds.append(im_ind_i)

        return torch.stack(gt_classes), torch.stack(gt_shifts_deltas), torch.stack(gt_centerness),\
            torch.cat(gt_instances_masks), torch.stack(gt_inds), torch.stack(im_inds), \
            torch.stack(fpn_levels)

    def generate_in_box_mask(self, boxes, masks, deltas, im_h, im_w, shifts_per_image):
        if self.center_sampling_radius > 0:
            if self.mask_center_sample:
                # select mass center as center sample point
                ys = torch.arange(0, im_h, device=masks.device)
                xs = torch.arange(0, im_w, device=masks.device)
                mask_pos_number = masks.sum([-2, -1]).clamp(min=1e-6)
                mask_weight_in_x = (masks * xs).sum([-2, -1])
                mask_weight_in_y = (masks * ys[:, None]).sum([-2, -1])
                center_x = mask_weight_in_x / mask_pos_number
                center_y = mask_weight_in_y / mask_pos_number
                centers = torch.stack((center_x, center_y), dim=1)
            else:
                centers = boxes.get_centers()

            is_in_boxes = []
            for stride, shifts_i in zip(self.fpn_strides, shifts_per_image):
                radius = stride * self.center_sampling_radius

                center_boxes = torch.cat((
                    torch.max(centers - radius, boxes.tensor[:, :2]),
                    torch.min(centers + radius, boxes.tensor[:, 2:]),
                ), dim=-1)
                center_deltas = self.shift2box_transform.get_deltas(
                    shifts_i, center_boxes.unsqueeze(1))
                is_in_boxes.append(center_deltas.min(dim=-1).values > 0)
            is_in_boxes = torch.cat(is_in_boxes, dim=1)
        else:
            # no center sampling, it will use all the locations within a ground-truth box
            is_in_boxes = deltas.min(dim=-1).values > 0

        return is_in_boxes

    def proposals_losses(self, gt_classes, gt_shifts_deltas, gt_centerness, gt_inds, im_inds,
                         pred_class_logits, pred_shift_deltas, pred_centerness,
                         pred_inst_params, fpn_levels, shifts):

        pred_class_logits, pred_shift_deltas, pred_centerness, pred_inst_params = \
            permute_all_to_N_HWA_K_and_concat(
                pred_class_logits, pred_shift_deltas, pred_centerness,
                pred_inst_params, self.num_gen_params, self.num_classes
            )  # Shapes: (N x R, K) and (N x R, 4), respectively.

        gt_classes = gt_classes.flatten()
        gt_shifts_deltas = gt_shifts_deltas.reshape(-1, 4)
        gt_centerness = gt_centerness.reshape(-1, 1)
        fpn_levels = fpn_levels.flatten()
        im_inds = im_inds.flatten()
        gt_inds = gt_inds.flatten()

        valid_idxs = gt_classes >= 0
        foreground_idxs = (gt_classes >= 0) & (gt_classes != self.num_classes)
        num_foreground = foreground_idxs.sum()

        gt_classes_target = torch.zeros_like(pred_class_logits)
        gt_classes_target[foreground_idxs, gt_classes[foreground_idxs]] = 1

        num_foreground = comm.all_reduce(num_foreground) / float(comm.get_world_size())
        num_foreground_centerness = gt_centerness[foreground_idxs].sum()
        num_targets = comm.all_reduce(num_foreground_centerness) / float(comm.get_world_size())

        # logits loss
        loss_cls = sigmoid_focal_loss_jit(
            pred_class_logits[valid_idxs],
            gt_classes_target[valid_idxs],
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        ) / max(1.0, num_foreground)

        # regression loss
        loss_box_reg = iou_loss(
            pred_shift_deltas[foreground_idxs],
            gt_shifts_deltas[foreground_idxs],
            gt_centerness[foreground_idxs],
            box_mode="ltrb",
            loss_type=self.iou_loss_type,
            reduction="sum",
            smooth=self.iou_smooth
        ) / max(1e-6, num_targets)

        # centerness loss
        loss_centerness = F.binary_cross_entropy_with_logits(
            pred_centerness[foreground_idxs],
            gt_centerness[foreground_idxs],
            reduction="sum",
        ) / max(1, num_foreground)

        proposals_losses = {
            "loss_cls": loss_cls,
            "loss_box_reg": loss_box_reg,
            "loss_centerness": loss_centerness
        }

        all_shifts = torch.cat([torch.cat(shift) for shift in shifts])
        proposals = Instances((0, 0))
        proposals.inst_parmas = pred_inst_params[foreground_idxs]
        proposals.fpn_levels = fpn_levels[foreground_idxs]
        proposals.shifts = all_shifts[foreground_idxs]
        proposals.gt_inds = gt_inds[foreground_idxs]
        proposals.im_inds = im_inds[foreground_idxs]

        # select_instances for saving memory
        if len(proposals):
            if self.topk_proposals_per_im != -1:
                proposals.gt_cls = gt_classes[foreground_idxs]
                proposals.pred_logits = pred_class_logits[foreground_idxs]
                proposals.pred_centerness = pred_centerness[foreground_idxs]
            proposals = self.select_instances(proposals)

        return proposals_losses, proposals

    def select_instances(self, proposals):
        """
        Implement random sample and intance-level sample for subsequent mask segmentation.
        These two method proposed by Condinst(conference version), Condinst(journal version)
        and Boxinst paper.

        Notes:
            1. Random sample method indicates instances are random select ``per batch``
               from foreground pixels. Though setting self.max_proposals to a positive
               number and self.topk_proposals_per_im to -1 value, random sample method
               is adopted. Default setting is 500 max proposals in Condinst (conference
               version).
            2. Instance-level sample indicates instances are selected ``per image`` depends
               on topk score of foreground pixels, but each instance at least generates
               one predicted mask. For one pixel, the score could utilize
               max score (pred_cls * pred_ctr) across all classes or score at the index
               of gt class label. Though setting self.max_proposals to -1  and
               self.topk_proposals_per_im to a positive number, instance-level sample method
               is adopted. Default setting is 64 proposals per image in Condinst (journal
               version) and Boxinst paper.
        """
        if self.max_proposals != -1 and len(proposals) > self.max_proposals:  # random per batch
            inds = torch.randperm(len(proposals), device=self.device)
            proposals = proposals[inds[:self.max_proposals]]
        elif self.topk_proposals_per_im != -1:  # instance-balanced sample per image
            instances_list_per_gt = []
            num_images = max(proposals.im_inds.unique()) + 1
            for i in range(num_images):
                instances_per_image = proposals[proposals.im_inds == i]
                if len(instances_per_image) == 0:
                    instances_list_per_gt.append(instances_per_image)
                    continue
                unique_gt_inds = instances_per_image.gt_inds.unique()
                num_instances_per_gt = max(int(self.topk_proposals_per_im / len(unique_gt_inds)), 1)
                for gt_ind in unique_gt_inds:
                    instances_per_gt = instances_per_image[instances_per_image.gt_inds == gt_ind]
                    if len(instances_per_gt) > num_instances_per_gt:
                        # balanced_with_max_score strategy
                        scores = instances_per_gt.pred_logits.sigmoid().max(dim=1)[0]
                        # balanced_with_class_score strategy
                        # gt_cls = instances_per_gt.gt_cls[0]
                        # scores = instances_per_gt.pred_logits.sigmoid()[:, gt_cls]
                        ctrness_pred = instances_per_gt.pred_centerness.sigmoid()[:, 0]
                        inds = (scores * ctrness_pred).topk(k=num_instances_per_gt, dim=0)[1]
                        instances_per_gt = instances_per_gt[inds]
                    instances_list_per_gt.append(instances_per_gt)
            proposals = Instances.cat(instances_list_per_gt)
        return proposals

    def instances_losses(self, gt_insts_mask, proposals, dummy_feature):
        """
        Arguments:
            gt_insts_mask (Tensor):
                Shape (N, mask_h, mask_w), where N = the number of GT instances per batch
                segmentation ground truth
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
            storing the loss. Used during training only. The dict key is: "loss_mask"
        """
        if len(proposals):
            gt_inds = proposals.gt_inds
            pred_instances_mask = proposals.pred_global_logits.sigmoid()
            gt_insts_mask = gt_insts_mask[gt_inds]. \
                unsqueeze(dim=1).to(dtype=pred_instances_mask.dtype)
            loss_mask = dice_loss(pred_instances_mask, gt_insts_mask).mean()
        else:
            loss_mask = dummy_feature.sum() * 0.
        return {
            "loss_mask": loss_mask
        }

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images,
                                        self.backbone.size_divisibility)
        return images

    def postprocess(self, results, output_height, output_width, resized_in_h, resized_in_w,
                    padded_im_h, padded_im_w):
        scale_x, scale_y = (output_width / resized_in_w, output_height / resized_in_h)
        # gather detection result to Instances
        results = Instances((output_height, output_width), **results.get_fields())
        # scale detection box results from resized_padded_image space to source image space and clip
        output_boxes = results.pred_boxes
        output_boxes.scale(scale_x, scale_y)
        output_boxes.clip(results.image_size)
        # filter empty detection in source image space
        results = results[output_boxes.nonempty()]
        if results.has("pred_global_logits"):
            mask_h, mask_w = results.pred_global_logits.shape[-2:]
            factor_h = padded_im_h // mask_h
            factor_w = padded_im_w // mask_w
            assert factor_h == factor_w
            factor = factor_h
            # aligned upsample instances mask to resized_padded_image shape
            pred_global_masks = aligned_bilinear(
                results.pred_global_logits.sigmoid(), factor
            )
            pred_global_masks = pred_global_masks[:, :, :resized_in_h, :resized_in_w]
            # scale mask from resized_image shape to source image shape
            # this is a inverse procedure of opencv or PIL interpolation
            # which align_corners is False
            pred_global_masks = F.interpolate(
                pred_global_masks,
                size=(output_height, output_width),
                mode="bilinear", align_corners=False
            )
            pred_global_masks = pred_global_masks[:, 0, :, :]
            # filter out the pred masks with low confidence score
            results.pred_masks = pred_global_masks > self.infer_mask_threshold

        return results

    def proposals_inference(self, box_cls, box_delta, box_center, box_param,
                            shifts, images):

        proposals = []
        box_cls = [permute_to_N_HWA_K(x, self.num_classes) for x in box_cls]
        box_delta = [permute_to_N_HWA_K(x, 4) for x in box_delta]
        box_center = [permute_to_N_HWA_K(x, 1) for x in box_center]
        box_param = [permute_to_N_HWA_K(x, self.num_gen_params) for x in box_param]
        # list[Tensor], one per level, each has shape (N, Hi x Wi, K or 4)

        for img_idx, shifts_per_image in enumerate(shifts):
            image_size = images.image_sizes[img_idx]
            box_cls_per_image = [
                box_cls_per_level[img_idx] for box_cls_per_level in box_cls
            ]
            box_reg_per_image = [
                box_reg_per_level[img_idx] for box_reg_per_level in box_delta
            ]
            box_ctr_per_image = [
                box_ctr_per_level[img_idx] for box_ctr_per_level in box_center
            ]
            box_param_per_image = [
                box_param_per_level[img_idx] for box_param_per_level in box_param
            ]
            fpn_level_per_image = [
                loc.new_ones(len(loc), dtype=torch.long) * level
                for level, loc in enumerate(shifts_per_image)
            ]

            proposals_per_image = self.inference_single_image(
                box_cls_per_image, box_reg_per_image, box_ctr_per_image,
                box_param_per_image, shifts_per_image, tuple(image_size),
                fpn_level_per_image, img_idx
            )
            proposals.append(proposals_per_image)

        proposals = Instances.cat(proposals)
        return proposals

    def inference_single_image(self, box_cls, box_delta, box_center, box_param,
                               shifts, image_size, fpn_levels, img_id):
        boxes_all = []
        scores_all = []
        class_idxs_all = []
        box_params_all = []
        shifts_all = []
        fpn_levels_all = []
        # Iterate over every feature level
        for box_cls_i, box_reg_i, box_ctr_i, box_param_i, shifts_i, fpn_level_i in zip(
                box_cls, box_delta, box_center, box_param, shifts, fpn_levels):

            box_cls_i = box_cls_i.flatten().sigmoid_()
            if self.thresh_with_centerness:
                box_ctr_i = box_ctr_i.expand((-1, self.num_classes)).flatten().sigmoid()
                box_cls_i = box_cls_i * box_ctr_i

            # Keep top k top scoring indices only.
            num_topk = min(self.topk_candidates, box_reg_i.shape[0])
            # torch.sort is actually faster than .topk (at least on GPUs)
            predicted_prob, topk_idxs = box_cls_i.sort(descending=True)
            predicted_prob = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:num_topk]

            # filter out the proposals with low confidence score
            keep_idxs = predicted_prob > self.score_threshold  # after topk
            predicted_prob = predicted_prob[keep_idxs]
            topk_idxs = topk_idxs[keep_idxs]

            shift_idxs = topk_idxs // self.num_classes
            classes_idxs = topk_idxs % self.num_classes

            box_reg_i = box_reg_i[shift_idxs]
            shifts_i = shifts_i[shift_idxs]
            fpn_level_i = fpn_level_i[shift_idxs]
            # predict boxes
            predicted_boxes = self.shift2box_transform.apply_deltas(
                box_reg_i, shifts_i
            )

            if not self.thresh_with_centerness:
                box_ctr_i = box_ctr_i.flatten().sigmoid_()[shift_idxs]
                predicted_prob = predicted_prob * box_ctr_i

            # instances conv params for predicted boxes
            box_param = box_param_i[shift_idxs]

            boxes_all.append(predicted_boxes)
            scores_all.append(torch.sqrt(predicted_prob))
            class_idxs_all.append(classes_idxs)
            box_params_all.append(box_param)
            shifts_all.append(shifts_i)
            fpn_levels_all.append(fpn_level_i)

        boxes_all, scores_all, class_idxs_all, box_params_all, shifts_all, fpn_levels_all = [
            cat(x) for x in
            [boxes_all, scores_all, class_idxs_all, box_params_all, shifts_all, fpn_levels_all]
        ]

        keep = generalized_batched_nms(
            boxes_all, scores_all, class_idxs_all,
            self.nms_threshold, nms_type=self.nms_type
        )
        keep = keep[:self.max_detections_per_image]

        im_inds = scores_all.new_ones(len(scores_all), dtype=torch.long) * img_id
        proposals_i = Instances(image_size)
        proposals_i.pred_boxes = Boxes(boxes_all[keep])
        proposals_i.scores = scores_all[keep]
        proposals_i.pred_classes = class_idxs_all[keep]

        proposals_i.inst_parmas = box_params_all[keep]
        proposals_i.fpn_levels = fpn_levels_all[keep]
        proposals_i.shifts = shifts_all[keep]
        proposals_i.im_inds = im_inds[keep]

        return proposals_i

    def _inference_for_ms_test(self, batched_inputs):
        """
        function used for multiscale test, will be refactor in the future.
        The same input with `forward` function.
        """
        assert not self.training, "inference mode with training=True"
        assert len(batched_inputs) == 1, "inference image number > 1"
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        features = [features[f] for f in self.in_features]
        box_cls, box_delta, box_center, box_param = self.head(features)
        shifts = self.shift_generator(features)
        proposals = self.proposals_inference(
            box_cls, box_delta, box_center, box_param, shifts, images
        )

        mask_input_feats = self.mask_branch(features)
        # get mask shift depend on downsample ratio
        mask_shift = shifts[0][self.mask_out_level_ind]
        proposals = self.dynamic_mask_head(
            mask_input_feats, self.mask_branch_out_stride,
            mask_shift, proposals
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


class DynamicMaskHead(nn.Module):

    def __init__(self, cfg):
        super(DynamicMaskHead, self).__init__()
        # fmt: off
        self.num_layers = cfg.MODEL.CONDINST.MASK_HEAD.NUM_LAYERS
        self.channels = cfg.MODEL.CONDINST.MASK_HEAD.HEAD_CHANNELS
        self.in_channels = cfg.MODEL.CONDINST.MASK_BRANCH.OUT_CHANNELS
        self.mask_out_stride = cfg.MODEL.CONDINST.MASK_OUT_STRIDE
        self.disable_rel_coords = cfg.MODEL.CONDINST.MASK_HEAD.DISABLE_REL_COORDS
        soi = [soi[1] for soi in cfg.MODEL.FCOS.OBJECT_SIZES_OF_INTEREST[:-1]]
        # fmt: on
        if self.disable_rel_coords:
            log_first_n(
                "WARNING",
                "Training CondInst without Coord",
                n=1
            )
        self.sizes_of_interest = torch.tensor(soi + [soi[-1] * 2])  # 64 128 256 512 1024

        weight_nums, bias_nums = [], []
        for layer_ind in range(self.num_layers):
            if layer_ind == 0:
                if not self.disable_rel_coords:
                    weight_nums.append((self.in_channels + 2) * self.channels)
                else:
                    weight_nums.append(self.in_channels * self.channels)
                bias_nums.append(self.channels)
            elif layer_ind == self.num_layers - 1:
                weight_nums.append(self.channels * 1)
                bias_nums.append(1)
            else:
                weight_nums.append(self.channels * self.channels)
                bias_nums.append(self.channels)

        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)
        assert len(weight_nums) == len(bias_nums)

    def forward(self, mask_feats, mask_feat_stride, mask_shift, proposals):

        n_inst = len(proposals)
        if n_inst > 0:
            im_inds = proposals.im_inds
            N, _, H, W = mask_feats.shape
            if not self.disable_rel_coords:
                instance_shifts = proposals.shifts
                relative_coords = instance_shifts.reshape(-1, 1, 2) - mask_shift.reshape(1, -1, 2)
                relative_coords = relative_coords.permute(0, 2, 1)
                # size of interest
                soi = self.sizes_of_interest[proposals.fpn_levels].to(relative_coords.device)
                relative_coords = relative_coords / soi.reshape(-1, 1, 1)  # norm
                mask_head_inputs = torch.cat([
                    relative_coords, mask_feats[im_inds].reshape(n_inst, self.in_channels, H * W)
                ], dim=1)
            else:
                mask_head_inputs = mask_feats[im_inds].reshape(n_inst, self.in_channels, H, W)
            mask_head_inputs = mask_head_inputs.reshape(1, -1, H, W)

            weights, biases = self.parse_dynamic_params(proposals.inst_parmas)
            mask_logits = self.mask_heads_forward(mask_head_inputs, weights, biases, n_inst)
            mask_logits = mask_logits.reshape(-1, 1, H, W)
            assert mask_feat_stride >= self.mask_out_stride
            assert mask_feat_stride % self.mask_out_stride == 0
            mask_logits = aligned_bilinear(
                mask_logits, int(mask_feat_stride / self.mask_out_stride)
            )
            proposals.pred_global_logits = mask_logits

        return proposals

    def mask_heads_forward(self, features, weights, biases, num_insts):
        """
        Using generated conv weights and biases, this func generates mask logits
        """
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
            )
            if i < n_layers - 1:
                x = F.relu(x)
        return x

    def parse_dynamic_params(self, params):
        """parse per-instances weights and biases

        Args:
            params (Tensor): per-location conv weights and biases,
                shape like (num_insts, sum(weight_nums)+sum(bias_nums))

        Returns:
            weight_splits (List[Tensor]): contains per-layer conv weights
                shape like (num_insts * output_channels, input_channels_per_inst , 1, 1)
            bias_splits (List[Tensor]): contains per-layer conv biases
                shape like (num_insts * output_channels, input_channels_per_inst , 1, 1)
        """
        assert params.dim() == 2
        assert params.shape[1] == sum(self.weight_nums) + sum(self.bias_nums)

        num_insts = params.shape[0]
        params_splits = list(torch.split_with_sizes(
            params, self.weight_nums + self.bias_nums, dim=1
        ))

        weight_splits = params_splits[:self.num_layers]
        bias_splits = params_splits[self.num_layers:]

        for layer_ind in range(self.num_layers):
            if layer_ind < self.num_layers - 1:
                weight_splits[layer_ind] = weight_splits[layer_ind].reshape(
                    num_insts * self.channels, -1, 1, 1
                )
                bias_splits[layer_ind] = bias_splits[layer_ind].reshape(num_insts * self.channels)
            else:
                weight_splits[layer_ind] = weight_splits[layer_ind].reshape(num_insts * 1, -1, 1, 1)
                bias_splits[layer_ind] = bias_splits[layer_ind].reshape(num_insts)

        return weight_splits, bias_splits


class CondInstHead(nn.Module):
    def __init__(self, cfg, num_gen_params, input_shape: List[ShapeSpec]):
        super(CondInstHead, self).__init__()
        # fmt: off
        in_channels = input_shape[0].channels
        num_classes = cfg.MODEL.FCOS.NUM_CLASSES
        num_convs = cfg.MODEL.FCOS.NUM_CONVS
        prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
        num_shifts = cfg.build_shift_generator(cfg, input_shape).num_cell_shifts
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.centerness_on_reg = cfg.MODEL.FCOS.CENTERNESS_ON_REG
        self.norm_reg_targets = cfg.MODEL.FCOS.NORM_REG_TARGETS
        # fmt: on
        assert len(set(num_shifts)) == 1, "using differenct num_shifts value is not supported"
        num_shifts = num_shifts[0]

        cls_subnet = []
        bbox_subnet = []
        for _ in range(num_convs):
            cls_subnet.append(
                nn.Conv2d(in_channels,
                          in_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1))
            cls_subnet.append(nn.GroupNorm(32, in_channels))
            cls_subnet.append(nn.ReLU())
            bbox_subnet.append(
                nn.Conv2d(in_channels,
                          in_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1))
            bbox_subnet.append(nn.GroupNorm(32, in_channels))
            bbox_subnet.append(nn.ReLU())

        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.bbox_subnet = nn.Sequential(*bbox_subnet)
        self.cls_score = nn.Conv2d(in_channels,
                                   num_shifts * num_classes,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
        self.bbox_pred = nn.Conv2d(in_channels,
                                   num_shifts * 4,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
        self.centerness = nn.Conv2d(in_channels,
                                    num_shifts * 1,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)
        self.inst_param_pred = nn.Conv2d(in_channels,
                                         num_gen_params,
                                         kernel_size=3,
                                         stride=1,
                                         padding=1)
        # Initialization
        for modules in [
            self.cls_subnet, self.bbox_subnet, self.cls_score,
            self.bbox_pred, self.centerness, self.inst_param_pred
        ]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)
                if isinstance(layer, nn.GroupNorm):
                    torch.nn.init.constant_(layer.weight, 1)
                    torch.nn.init.constant_(layer.bias, 0)

        # Use prior in model initialization to improve stability
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_score.bias, bias_value)

        self.scales = nn.ModuleList(
            [Scale(init_value=1.0) for _ in range(len(self.fpn_strides))])

    def forward(self, features):
        logits = []
        bbox_reg = []
        centerness = []
        inst_params = []
        for level, feature in enumerate(features):
            cls_subnet = self.cls_subnet(feature)
            bbox_subnet = self.bbox_subnet(feature)

            logits.append(self.cls_score(cls_subnet))
            if self.centerness_on_reg:
                centerness.append(self.centerness(bbox_subnet))
            else:
                centerness.append(self.centerness(cls_subnet))

            bbox_pred = self.scales[level](self.bbox_pred(bbox_subnet))
            if self.norm_reg_targets:
                bbox_reg.append(F.relu(bbox_pred) * self.fpn_strides[level])
            else:
                bbox_reg.append(torch.exp(bbox_pred))
            inst_params.append(self.inst_param_pred(bbox_subnet))
        return logits, bbox_reg, centerness, inst_params


class MaskBranch(nn.Module):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super(MaskBranch, self).__init__()
        # fmt: off
        norm = cfg.MODEL.CONDINST.MASK_BRANCH.NORM
        in_features = cfg.MODEL.CONDINST.MASK_BRANCH.IN_FEATURES
        num_convs = cfg.MODEL.CONDINST.MASK_BRANCH.NUM_CONVS
        channels = cfg.MODEL.CONDINST.MASK_BRANCH.BRANCH_CHANNELS
        fpn_out_features = cfg.MODEL.FCOS.IN_FEATURES
        self.num_outputs = cfg.MODEL.CONDINST.MASK_BRANCH.OUT_CHANNELS
        # fmt: on

        self.in_features_inds = [fpn_out_features.index(in_fea_key)
                                 for in_fea_key in in_features]
        self.out_stride = input_shape[in_features[0]].stride

        self.refine = nn.ModuleList()
        for in_feature in in_features:
            refine_i = [nn.Conv2d(input_shape[in_feature].channels,
                                  channels,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1,
                                  bias=(norm is None)
                                  ),
                        get_norm(norm, channels),
                        nn.ReLU(inplace=True)]
            self.refine.append(nn.Sequential(*refine_i))

        mask_subnet = []
        for _ in range(num_convs):
            mask_subnet.append(nn.Conv2d(channels, channels,
                                         kernel_size=3,
                                         stride=1,
                                         padding=1,
                                         bias=(norm is None)
                                         ))
            mask_subnet.append(get_norm(norm, channels))
            mask_subnet.append(nn.ReLU(inplace=True))

        mask_subnet.append(nn.Conv2d(
            channels, max(self.num_outputs, 1), 1
        ))
        self.mask_subnet = nn.Sequential(*mask_subnet)

        # Initialization
        for modules in [
            self.refine, self.mask_subnet
        ]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.kaiming_uniform_(layer.weight, a=1)

    def forward(self, features):
        for i, f_i in enumerate(self.in_features_inds):
            if i == 0:
                x = self.refine[i](features[f_i])
            else:
                x_p = self.refine[i](features[f_i])
                target_h, target_w = x.shape[2:]
                h, w = x_p.shape[2:]
                factor_h = target_h // h
                x_p = aligned_bilinear(x_p, factor_h)
                x = x + x_p
        mask_feats = self.mask_subnet(x)
        return mask_feats
