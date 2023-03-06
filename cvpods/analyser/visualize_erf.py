#!/usr/bin/env python3
# Copyright Megvii Inc. All Rights Reserved

# A script to visualize the ERF(effective receptive field).
# Scaling Up Your Kernels to 31x31: Revisiting Large Kernel Design in CNNs (https://arxiv.org/abs/2203.06717)
# Github source: https://github.com/DingXiaoH/RepLKNet-pytorch  MIT License

import os
import argparse
import numpy as np
import torch
from torch import nn
from typing import List

import timm
from timm.utils import AverageMeter

FEATURE_LIST = []


def make_parser():
    parser = argparse.ArgumentParser('Script for visualizing the ERF', add_help=False)
    parser.add_argument('--model', default='resnet50', type=str, help='model name')
    parser.add_argument('--weights', default=None, type=str, help='path to weights file.')
    parser.add_argument('--pretrained', action="store_true", help='whether to use pretrained weights')
    parser.add_argument('--num_images', default=50, type=int, help='num of images to use')
    parser.add_argument('--name_list', default=["conv1"], nargs='+', type=str, help='layer name of the modules to analyze effective receptive field')
    parser.add_argument('--save_path', default='./', type=str, help='directory to save the heatmap')
    parser.add_argument('--threshold', default=(0.2, 0.3, 0.5, 0.99), nargs='+', type=float, help='thresh holds for get high-contribution area')
    return parser


def get_rectangle(data, thresh: float):
    h, w = data.shape
    all_sum = np.sum(data)
    for i in range(1, h // 2):
        selected_area = data[h // 2 - i:h // 2 + 1 + i, w // 2 - i:w // 2 + 1 + i]
        area_sum = np.sum(selected_area)
        if area_sum / all_sum > thresh:
            return i * 2 + 1, (i * 2 + 1) / h * (i * 2 + 1) / w


def plot_erf_img(layer_name, visual_data, save_name):
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.heatmap(
        visual_data,
        xticklabels=False,
        yticklabels=False, cmap='RdYlGn',
        center=0, annot=False, cbar=True,
        ax=None, annot_kws={"size": 24}, fmt='.2f'
    )
    plt.title(f'Effective Receptive Field of model.{layer_name}')
    plt.savefig(save_name)
    plt.close()


def analyze_erf(layer_name, data, thresh_list=[0.2, 0.3, 0.5, 0.99], save_path=''):
    """

    Args:
        name: name of the layer
        data: input data 
        thresh_list: 
        save_path: directory to save the heatmap
    """
    data = np.log10(data + 1)   # the scores differ in magnitude. take the logarithm for better readability
    data = data / np.max(data)  # rescale to [0,1] for the comparability among models
    save_name = os.path.join(save_path, 'erf_' + layer_name + '.png')
    plot_erf_img(layer_name, data, save_name)

    if thresh_list is None or not thresh_list:
        return

    side_length_list = []
    area_ratio_list = []

    for thresh in thresh_list:
        side_length, area_ratio = get_rectangle(data, thresh)
        side_length_list.append(side_length)
        area_ratio_list.append(area_ratio)

    print(f"\nERF analysis of model.{layer_name}:")
    print("\tThreshold\t"+'\t'.join(map(str, thresh_list)))
    print("\tArea Ratio\t"+'\t'.join(map(str, np.round(area_ratio_list, 4))))
    print("\tSide Length\t"+'\t'.join(map(str, side_length_list)))
    print(f"Check the the visualized erf of model.{layer_name} at {save_name}")


def get_input_grad(model, samples):
    global FEATURE_LIST
    FEATURE_LIST.clear()
    model(samples)

    grad_map_list = []
    for feat in FEATURE_LIST:
        feat_size = feat.size()
        central_point = torch.nn.functional.relu(feat[:, :, feat_size[2] // 2, feat_size[3] // 2]).sum()
        grad = torch.autograd.grad(central_point, samples, retain_graph=True)[0]
        grad = torch.nn.functional.relu(grad)
        grad_map = grad.sum((0, 1)).cpu().numpy()
        grad_map_list.append(grad_map)
    return grad_map_list


def get_output_hook(model, input, output):
    global FEATURE_LIST
    FEATURE_LIST.append(output)


def visualize_erf(
    model: nn.Module,
    layer_names: List[str],
    dataloader,
    num_images: int = 10,
    thresh_list: List[float] = [0.2, 0.3, 0.5, 0.99],
    save_path: str = ""
):
    """visualize the effective receptive field of specified layers in `layer_names`.

    Args:
        model (nn.Module): 
        layer_names (List[str]): name of model layers to be analyzed.
        dataloader : data provider to analyze the erf.
        num_images (int, optional): total number of images to analyze. Defaults to 10.
        thresh_list (List[float], optional): Defaults to [0.2, 0.3, 0.5, 0.99].
        save_path (str, optional): path to save the img. Defaults to current dir. 
    """

    for name, module in model.named_modules():
        if name in layer_names:
            module.register_forward_hook(get_output_hook)

    num_layers = len(layer_names)
    meters = [AverageMeter() for i in range(num_layers)]

    for sample_idx, samples in enumerate(dataloader, 1):
        if torch.cuda.is_available():
            samples = samples.cuda(non_blocking=True)
        samples.requires_grad = True

        contribution_scores = get_input_grad(model, samples)

        if np.isnan(np.sum(contribution_scores)):
            continue
        else:
            assert len(contribution_scores) == num_layers
            [meters[i].update(contribution_scores[i]) for i in range(num_layers)]

        if sample_idx == num_images:
            break

    for name, meter in zip(layer_names, meters):
        analyze_erf(name, meter.avg, thresh_list=thresh_list, save_path=save_path)


def main():
    # transform: resize to 1024x1024
    args = make_parser().parse_args()
    print("args:", args)
    kwargs = {}
    if args.weights is None:
        kwargs["pretrained"] = args.pretrained
    else:
        kwargs["checkpoint_path"] = args.weights
    model = timm.create_model(args.model, **kwargs)

    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    data_loader = [torch.rand(1, 3, 1024, 1024) for i in range(args.num_images)]
    visualize_erf(
        model, args.name_list, data_loader,
        num_images=args.num_images,
        thresh_list=args.threshold,
        save_path=args.save_path
    )


if __name__ == '__main__':
    main()
