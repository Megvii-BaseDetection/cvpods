#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# This file has been modified by Megvii ("Megvii Modifications").
# All Megvii Modifications are Copyright (C) 2019-2021 Megvii Inc. All rights reserved.
import importlib
import importlib.util
import os
import random
import socket
import sys
from datetime import datetime
from loguru import logger

import numpy as np

import torch

__all__ = [
    "seed_all_rng",
    "setup_environment",
    "setup_custom_environment",
    "get_host_ip",
    "TORCH_VERSION",
]

TORCH_VERSION = tuple(int(x) for x in torch.__version__.split(".")[:2])


def seed_all_rng(seed=None):
    """
    Set the random seed for the RNG in torch, numpy and python.

    Args:
        seed (int): if None, will use a strong random seed.

    Returns:
        seed (int): used seed value.
    """
    if seed is None:
        seed = (
            os.getpid()
            + int(datetime.now().strftime("%S%f"))
            + int.from_bytes(os.urandom(2), "big")
        )
        logger.info("Using a generated random seed {}".format(seed))
    np.random.seed(seed)
    torch.set_rng_state(torch.manual_seed(seed).get_state())
    random.seed(seed)
    return seed


# from https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
def _import_file(module_name, file_path, make_importable=False):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if make_importable:
        sys.modules[module_name] = module
    return module


def _configure_libraries():
    """
    Configurations for some libraries.
    """
    # An environment option to disable `import cv2` globally,
    # in case it leads to negative performance impact
    disable_cv2 = int(os.environ.get("cvpods_DISABLE_CV2", False))
    if disable_cv2:
        sys.modules["cv2"] = None
    else:
        # Disable opencl in opencv since its interaction with cuda often has negative effects
        # This envvar is supported after OpenCV 3.4.0
        os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"
        try:
            import cv2

            if int(cv2.__version__.split(".")[0]) >= 3:
                cv2.ocl.setUseOpenCL(False)
        except ImportError:
            pass


_ENV_SETUP_DONE = False


def setup_environment():
    """Perform environment setup work. The default setup is a no-op, but this
    function allows the user to specify a Python source file or a module in
    the $cvpods_ENV_MODULE environment variable, that performs
    custom setup work that may be necessary to their computing environment.
    """
    global _ENV_SETUP_DONE
    if _ENV_SETUP_DONE:
        return
    _ENV_SETUP_DONE = True

    _configure_libraries()

    custom_module_path = os.environ.get("cvpods_ENV_MODULE")

    if custom_module_path:
        setup_custom_environment(custom_module_path)
    else:
        # The default setup is a no-op
        pass


def setup_custom_environment(custom_module):
    """
    Load custom environment setup by importing a Python source file or a
    module, and run the setup function.
    """
    if custom_module.endswith(".py"):
        module = _import_file("cvpods.utils.env.env.custom_module", custom_module)
    else:
        module = importlib.import_module(custom_module)
    assert hasattr(module, "setup_environment") and callable(module.setup_environment), (
        "Custom environment module defined in {} does not have the "
        "required callable attribute 'setup_environment'."
    ).format(custom_module)
    module.setup_environment()


def get_host_ip():
    """
    Get host IP value. A dict contains IP information will be returned.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # 10.255.255.255 is a local ip address, no need to be reachable
        host, port = "10.255.255.255", 1
        s.connect((host, port))
        IP = s.getsockname()[0]
    except Exception:
        # try by netiface
        ip_dict = get_hostip_by_netiface()
        if not ip_dict:
            IP = "127.0.0.1"
        else:
            return ip_dict
    finally:
        s.close()
    return {"IP": IP}


def get_hostip_by_netiface():
    try:
        import netifaces
    except ImportError:
        return {}

    no_ip_string = "No IP addr"
    no_addr = [{"addr": no_ip_string}]
    ip_dict = {}

    for interface in netifaces.interfaces():
        # skip local
        if interface == "lo":
            continue
        net_dict = netifaces.ifaddresses(interface).setdefault(netifaces.AF_INET, no_addr)
        addr = [i["addr"] for i in net_dict]
        if len(addr) == 1:
            addr = addr[0]
        if addr == no_ip_string:
            continue
        ip_dict["IP of " + interface] = addr

    return ip_dict
