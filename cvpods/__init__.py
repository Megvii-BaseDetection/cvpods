# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from .utils import setup_environment, set_megfile

setup_environment()
set_megfile()

# This line will be programatically read/write by setup.py.
# Leave them at the bottom of this file and don't touch them.
__version__ = "0.1"
