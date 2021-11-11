#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified by BaseDetection group. All Rights Reserved

import glob
import os
import subprocess
from os import path
from setuptools import find_packages, setup

import torch
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 3], "Requires PyTorch >= 1.3"


def get_cuda_version():
    cuda_version = ""
    try:
        nvcc = get_command_path("nvcc")
        nvcc = subprocess.check_output(
            "'{}' -V | grep 'Cuda compilation tools'".format(nvcc),
            shell=True
        )
        nvcc = nvcc.decode("utf-8").strip()
        version = "".join(nvcc.split()[-1].split(".")[:2])
        version = version[1:]
    except subprocess.SubprocessError:
        nvcc = "Not Available"
        version = 'none'
    finally:
        cuda_version = version

    return cuda_version


def get_command_path(command_name):
    """
    Get path of given command.

    NOTE: This function only works on linux platform.
    """
    with open(os.devnull, "w") as devnull:
        command_path = subprocess.check_output(
            ["which", command_name], stderr=devnull
        ).decode().rstrip('\r\n')
    return command_path


def get_version():
    init_py_path = path.join(path.abspath(path.dirname(__file__)), "cvpods", "__init__.py")
    init_py = open(init_py_path, "r").readlines()
    version_line = [line.strip() for line in init_py if line.startswith("__version__")][0]
    version = version_line.split("=")[-1].strip().strip("'\"")

    # The following is used to build release packages.
    # Users should never use it.
    suffix = os.getenv("PODS_VERSION_SUFFIX", "")
    version = version + suffix
    cuda_version = get_cuda_version()
    version += "+cu{}torch{}".format(cuda_version, "".join([str(x) for x in torch_ver]))
    if os.getenv("BUILD_NIGHTLY", "0") == "1":
        from datetime import datetime

        date_str = datetime.today().strftime("%y%m%d")
        version = version + ".dev" + date_str

        new_init_py = [line for line in init_py if not line.startswith("__version__")]
        new_init_py.append('__version__ = "{}"\n'.format(version))
        with open(init_py_path, "w") as f:
            f.write("".join(new_init_py))
    return version


def get_extensions():
    this_dir = path.dirname(path.abspath(__file__))
    extensions_dir = path.join(this_dir, "cvpods", "layers", "csrc")

    main_source = path.join(extensions_dir, "vision.cpp")
    sources = glob.glob(path.join(extensions_dir, "**", "*.cpp"))
    source_cuda = glob.glob(path.join(extensions_dir, "**", "*.cu")) + glob.glob(
        path.join(extensions_dir, "*.cu")
    )

    sources = [main_source] + sources
    extension = CppExtension

    extra_compile_args = {"cxx": []}
    define_macros = []

    if (
        torch.cuda.is_available() and CUDA_HOME is not None and os.path.isdir(CUDA_HOME)
    ) or os.getenv("FORCE_CUDA", "0") == "1":
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]

        # It's better if pytorch can do this by default ..
        CC = os.environ.get("CC", None)
        if CC is not None:
            extra_compile_args["nvcc"].append("-ccbin={}".format(CC))

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "cvpods._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


def build_cvpods_script():
    cur_dir = os.getcwd()
    head = ("#!/bin/bash\n")
    with open("tools/pods_train", "w") as pods_train:
        pods_train.write(head + f"python3 {os.path.join(cur_dir, 'tools', 'train_net.py')} $@")

    with open("tools/pods_test", "w") as pods_test:
        pods_test.write(head + f"python3 {os.path.join(cur_dir, 'tools', 'test_net.py')} $@")

    with open("tools/pods_debug", "w") as pods_debug:
        pods_debug.write(head + f"python3 {os.path.join(cur_dir, 'tools', 'debug_net.py')} $@")


if __name__ == "__main__":
    build_cvpods_script()
    setup(
        name="cvpods",
        version=get_version(),
        author="BaseDetection",
        description="cvpods is BaseDetection's research "
        "platform for object detection and segmentation based on cvpods.",
        packages=find_packages(exclude=("configs", "tests")),
        python_requires=">=3.6",
        install_requires=[
            "cython",
            "torch",
            "torchvision",
            # Do not add opencv here. Just like pytorch, user should install
            # opencv themselves, preferrably by OS's package manager, or by
            # choosing the proper pypi package name at https://github.com/skvark/opencv-python
            "termcolor>=1.1",
            "colorama",
            "Pillow>=7.1",  # or use pillow-simd for better performance
            "opencv-python",
            "tabulate",
            "cloudpickle",
            "matplotlib",
            "loguru",
            "timm",
            "megfile",
            "tqdm>4.29.0",
            "tensorboard",
            "pycocotools>=2.0.2",  # corresponds to https://github.com/ppwwyyxx/cocoapi
            "future",  # used by caffe2
            "pydot",  # used to save caffe2 SVGs
            "portalocker",
            "easydict",
            "appdirs",
            "seaborn",
            "pandas",
            "lvis",
            "sklearn",
        ],
        extras_require={
            "all": [
                "shapely",
                "psutil",
                "hydra-core",
                "panopticapi @ https://github.com/cocodataset/panopticapi/archive/master.zip",
                "cityscapesscripts",
            ],
        },
        ext_modules=get_extensions(),
        cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
        scripts=[
            "tools/pods_train",
            "tools/pods_test",
            "tools/pods_debug",
        ],
    )
