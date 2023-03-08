#!/usr/bin/env python3
# Copyright Megvii Inc. All Rights Reserved

import argparse
import os

import matplotlib.pyplot as plt

import torch
import torch.utils.model_zoo as model_zoo

WEIGHTS_URL = {
    "supervised": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
    "byol": "https://download.openmmlab.com/mmselfsup/byol/byol_resnet50_8xb32-accum16-coslr-200e_in1k_20220225-5c8b2c2e.pth",
    "moco": "https://download.openmmlab.com/mmselfsup/moco/mocov2_resnet50_8xb32-coslr-200e_in1k_20220225-89e03af4.pth",
    "simclr": "https://download.openmmlab.com/mmselfsup/simclr/simclr_resnet50_8xb32-coslr-200e_in1k_20220428-46ef6bb9.pth"

}


def make_parser():
    parser = argparse.ArgumentParser('Script for analyzing the model weights', add_help=False)
    parser.add_argument(
        '--anchor_model', default='moco', nargs="+", type=str, help='choose the model as an anchor.'
    )
    parser.add_argument(
        '--target_model', default=['supervised', 'byol'], nargs="+", type=tuple,
        choices=["byol", "moco", "simclr"], help='choose the models to analyze.'
    )
    parser.add_argument(
        '--name_filter', default='conv2', type=str, help='to get the modules you want to analyze.'
    )
    parser.add_argument(
        '--statistics_name', default='norm', type=str, help='the statistics name for analyzing.'
    )
    parser.add_argument(
        '--save_path', default='./', type=str, help='directory to save the results.'
    )
    return parser


def load_state_dict(model_name):
    url = WEIGHTS_URL.get(model_name, None)
    assert url is not None, f"model {model_name} is not supported."

    ckpt = model_zoo.load_url(url, progress=False, map_location='cpu')
    if 'model' in ckpt.keys():
        ckpt = ckpt['model']
    elif 'state_dict' in ckpt.keys():
        ckpt = ckpt['state_dict']
    return ckpt


def get_statistics(anchor, target, stats, filter_name=None):
    res = {'name': [], 'anchor': [], 'target': []}
    for i in anchor.keys():
        if 'bias' in i or i not in target.keys() or anchor[i].dtype is not torch.float:
            continue
        if filter_name is not None:
            if filter_name not in i:
                continue
        res['name'].append(i)
        if stats == 'mean':
            res['anchor'].append(anchor[i].detach().mean().numpy())
            res['target'].append(target[i].detach().mean().numpy())
        elif stats == 'std':
            res['anchor'].append(anchor[i].detach().std().numpy())
            res['target'].append(target[i].detach().std().numpy())
        elif stats == 'max':
            res['anchor'].append(anchor[i].detach().abs().max().numpy())
            res['target'].append(target[i].detach().abs().max().numpy())
        elif stats == 'norm':
            res['anchor'].append(anchor[i].detach().norm().numpy())
            res['target'].append(target[i].detach().norm().numpy())
    return res


def plot_figure(results_list, args):
    colors = ['b', 'r']
    markers = ['^', 'v']

    num_figs = len(args.target_model)
    _, subs = plt.subplots(1, num_figs, sharey=True)
    anchor_name = args.anchor_model
    for i in range(num_figs):
        res = results_list[i]
        x = range(len(res['name']))
        subs[i].plot(x, res['anchor'], color=colors[0], marker=markers[0])
        subs[i].plot(x, res['target'], color=colors[1], marker=markers[1])
        subs[i].legend([anchor_name, args.target_model[i]])
    save_path = os.path.join(args.save_path, 'analyze_' + args.statistics_name +'.png')
    plt.savefig(save_path)
    print(f"save image to {save_path}.")
    plt.close()


def demo():
    args = make_parser().parse_args()
    modelA = load_state_dict(args.anchor_model)
    results_list = []
    for target_model in args.target_model:
        modelB = load_state_dict(target_model)
        results_list.append(get_statistics(modelA, modelB, args.statistics_name, args.name_filter))
    plot_figure(results_list, args)


if __name__ == '__main__':
    import seaborn as sns
    sns.set_style("whitegrid")
    demo()
