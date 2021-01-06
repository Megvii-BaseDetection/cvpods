"""
Sparse R-CNN
"""
import torch
import torch.nn.functional as F
from torch import nn

import cvpods.structures.boxes as box_ops
from cvpods.layers import ShapeSpec
from cvpods.modeling import detector_postprocess
from cvpods.modeling.meta_arch.detr import PostProcess
from cvpods.structures import Boxes, ImageList, Instances

from head import DynamicHead
from loss import HungarianMatcher, SetCriterion


class SparseRCNN(nn.Module):
    def __init__(self, cfg):
        super(SparseRCNN, self).__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.num_classes = cfg.MODEL.SPARSE_RCNN.NUM_CLASSES
        self.num_proposals = cfg.MODEL.SPARSE_RCNN.NUM_PROPOSALS
        self.hidden_dim = cfg.MODEL.SPARSE_RCNN.HIDDEN_DIM
        self.num_heads = cfg.MODEL.SPARSE_RCNN.NUM_HEADS

        # Build Backbone
        self.backbone = cfg.build_backbone(
            cfg, input_shape=ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))
        )

        # Build Proposals.
        self.init_proposal_features = nn.Embedding(self.num_proposals, self.hidden_dim)
        self.init_proposal_boxes = nn.Embedding(self.num_proposals, 4)
        nn.init.constant_(self.init_proposal_boxes.weight[:, :2], 0.5)
        nn.init.constant_(self.init_proposal_boxes.weight[:, 2:], 1.0)

        # Build head.
        self.head = DynamicHead(cfg, self.backbone.output_shape())

        # Loss parameters.
        class_weight = cfg.MODEL.SPARSE_RCNN.CLASS_WEIGHT
        giou_weight = cfg.MODEL.SPARSE_RCNN.GIOU_WEIGHT
        l1_weight = cfg.MODEL.SPARSE_RCNN.L1_WEIGHT
        self.deep_supervision = cfg.MODEL.SPARSE_RCNN.DEEP_SUPERVISION
        self.use_focal = cfg.MODEL.SPARSE_RCNN.USE_FOCAL

        self.weight_dict = {
            "loss_ce": class_weight,
            "loss_bbox": l1_weight,
            "loss_giou": giou_weight,
        }

        if self.deep_supervision:
            self.aux_weight_dict = {}
            for i in range(self.num_heads - 1):
                self.aux_weight_dict.update({k + f"_{i}": v for k, v in self.weight_dict.items()})
            self.weight_dict.update(self.aux_weight_dict)

        losses = ["labels", "boxes", "cardinality"]

        matcher = HungarianMatcher(
            cfg=cfg,
            cost_class=class_weight,
            cost_bbox=l1_weight,
            cost_giou=giou_weight,
            use_focal=self.use_focal
        )

        self.criterion = SetCriterion(
            cfg,
            matcher=matcher,
            weight_dict=self.weight_dict,
            losses=losses,
        )

        self.post_processors = {"bbox": PostProcess()}

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)

        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        self.to(self.device)

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                * image: Tensor, image in (C, H, W) format.
                * instances: Instances
                Other information that's included in the original dicts, such as:
                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        """
        images, images_whwh = self.preprocess_image(batched_inputs)

        # Feature Extraction.
        src = self.backbone(images.tensor)
        features = list()
        for f in self.in_features:
            feature = src[f]
            features.append(feature)

        # Prepare Proposals.
        proposal_boxes = self.init_proposal_boxes.weight.clone()
        proposal_boxes = box_ops.box_cxcywh_to_xyxy(proposal_boxes)
        proposal_boxes = proposal_boxes[None] * images_whwh[:, None, :]

        # Prediction.
        outputs_class, outputs_coord = self.head(
            features, proposal_boxes, self.init_proposal_features.weight
        )
        output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}

        if self.training:
            targets = self.convert_anno_format(batched_inputs)
            if self.deep_supervision:
                output["aux_outputs"] = [
                    {"pred_logits": a, "pred_boxes": b}
                    for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
                ]
            loss_dict = self.criterion(output, targets)
            for k, v in loss_dict.items():
                loss_dict[k] = v * self.weight_dict[k] if k in self.weight_dict else v
            return loss_dict

        else:
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            results = self.inference(box_cls, box_pred, images.image_sizes)

            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                    results, batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})

            return processed_results

    def inference(self, box_cls, box_pred, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_proposals, K).
                The tensor predicts the classification probability for each proposal.
            box_pred (Tensor): tensors of shape (batch_size, num_proposals, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every proposal
            image_sizes (List[torch.Size]): the input image sizes
        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []

        if self.use_focal:
            scores = torch.sigmoid(box_cls)
            labels = torch.arange(self.num_classes, device=self.device). \
                unsqueeze(0).repeat(self.num_proposals, 1).flatten(0, 1)

            for i, (scores_per_image, box_pred_per_image, image_size) in enumerate(zip(
                    scores, box_pred, image_sizes
            )):
                result = Instances(image_size)
                scores_per_image, topk_indices = scores_per_image.flatten(0, 1).topk(
                    self.num_proposals, sorted=False)
                labels_per_image = labels[topk_indices]
                box_pred_per_image = box_pred_per_image.view(-1, 1, 4).repeat(
                    1, self.num_classes, 1).view(-1, 4)
                box_pred_per_image = box_pred_per_image[topk_indices]

                result.pred_boxes = Boxes(box_pred_per_image)
                result.scores = scores_per_image
                result.pred_classes = labels_per_image
                results.append(result)

        else:
            scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

            for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) \
                    in enumerate(zip(scores, labels, box_pred, image_sizes)):
                result = Instances(image_size)
                result.pred_boxes = Boxes(box_pred_per_image)
                result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])

                result.scores = scores_per_image
                result.pred_classes = labels_per_image
                results.append(result)

        return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].float().to(self.device) for x in batched_inputs]
        images = [self.normalizer(img) for img in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)

        images_whwh = list()
        for bi in batched_inputs:
            h, w = bi["image"].shape[-2:]
            images_whwh.append(torch.tensor([w, h, w, h], dtype=torch.float32, device=self.device))
        images_whwh = torch.stack(images_whwh)

        return images, images_whwh

    def convert_anno_format(self, batched_inputs):
        targets = []
        for bi in batched_inputs:
            target = {}
            assert bi["image"].shape[-2:] == bi["instances"].image_size
            h, w = bi["image"].shape[-2:]
            image_size_xyxy = torch.tensor([w, h, w, h], dtype=torch.float32)
            boxes = bi["instances"].gt_boxes.tensor / image_size_xyxy
            target["labels"] = bi["instances"].gt_classes.to(self.device)
            target["boxes"] = box_ops.box_xyxy_to_cxcywh(boxes).to(self.device)
            target["boxes_xyxy"] = bi["instances"].gt_boxes.tensor.to(self.device)
            target["area"] = bi["instances"].gt_boxes.area().to(self.device)
            target["image_size_xyxy"] = image_size_xyxy.to(self.device)
            image_size_xyxy_tgt = image_size_xyxy.unsqueeze(0).repeat(len(boxes), 1)
            target["image_size_xyxy_tgt"] = image_size_xyxy_tgt.to(self.device)
            if hasattr(bi["instances"], "gt_masks"):
                target["masks"] = bi["instances"].gt_masks
            target["iscrowd"] = torch.zeros_like(target["labels"], device=self.device)
            target["orig_size"] = torch.tensor([bi["height"], bi["width"]], device=self.device)
            target["size"] = torch.tensor([h, w], device=self.device)
            target["image_id"] = torch.tensor(bi["image_id"], device=self.device)
            targets.append(target)

        return targets
