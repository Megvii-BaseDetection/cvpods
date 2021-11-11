#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# This file has been modified by Megvii ("Megvii Modifications").
# All Megvii Modifications are Copyright (C) 2019-2021 Megvii Inc. All rights reserved.

import functools
import os
from typing import Optional
import megfile
import portalocker

from ..compat_wrapper import deprecated_func
from .download import download

__all__ = ["PathManager", "get_cache_dir", "file_lock", "ensure_dir", "set_megfile"]


def ensure_dir(path: str):
    """create directories if *path* does not exist"""""
    if not megfile.smart_isdir(path):
        megfile.smart_makedirs(path, exist_ok=True)


def get_cache_dir(cache_dir: Optional[str] = None) -> str:
    """
    Returns a default directory to cache static files
    (usually downloaded from Internet), if None is provided.

    Args:
        cache_dir (None or str): if not None, will be returned as is.
            If None, returns the default cache directory as:

        1) $CVPODS_CACHE, if set
        2) otherwise ~/.torch/cvpods_cache
    """
    if cache_dir is None:
        cache_dir = os.path.expanduser(
            os.getenv("CVPODS_CACHE", "~/.torch/cvpods_cache")
        )
    return cache_dir


def file_lock(path: str):  # type: ignore
    """
    A file lock. Once entered, it is guaranteed that no one else holds the
    same lock. Others trying to enter the lock will block for 30 minutes and
    raise an exception.

    This is useful to make sure workers don't cache files to the same location.

    Args:
        path (str): a path to be locked. This function will create a lock named
            `path + ".lock"`

    Examples:

    >>> filename = "/path/to/file"
    >>> with file_lock(filename):
            if not os.path.isfile(filename):
                do_create_file()
    """
    dirname = os.path.dirname(path)
    try:
        os.makedirs(dirname, exist_ok=True)
    except OSError:
        # makedir is not atomic. Exceptions can happen when multiple workers try
        # to create the same dir, despite exist_ok=True.
        # When this happens, we assume the dir is created and proceed to creating
        # the lock. If failed to create the directory, the next line will raise
        # exceptions.
        pass
    return portalocker.Lock(path + ".lock", timeout=1800)  # type: ignore


def cache_file(f):
    """wrapper of caching file logic for s3/http/https protocols in megfile"""

    @functools.wraps(f)
    def inner_func(path, *args, **kwargs):
        cache_dir = get_cache_dir()
        protocol, path_without_protocol = megfile.SmartPath._extract_protocol(path)
        local_path = os.path.join(cache_dir, path_without_protocol)
        if megfile.smart_exists(local_path):  # already cached
            return megfile.smart_open(local_path, *args, **kwargs)

        # caching logic
        if protocol == "s3":
            with file_lock(local_path):
                megfile.s3_download(path, local_path)
        elif protocol == "http" or protocol == "https":
            with file_lock(local_path):
                if not isinstance(path, str):
                    path = path.abspath()
                download(
                    path, os.path.dirname(local_path),
                    filename=os.path.basename(local_path)
                )

        return megfile.smart_open(local_path, *args, **kwargs)

    return inner_func


def set_megfile():
    import cvpods.checkpoint.catalog  # register d2 and catelog  # noqa
    megfile.HttpPath.open = cache_file(megfile.HttpPath.open)
    megfile.HttpsPath.open = cache_file(megfile.HttpsPath.open)


set_megfile()


# PathManager will be deprecated in the future.
class PathManager:

    @staticmethod
    @deprecated_func("use megfile.smart_open instead")
    def open(path: str, mode: str = "r"):
        """
        Open a stream to a URI, similar to the built-in `open`.

        Args:
            path (str): A URI supported by this PathHandler

        Returns:
            file: a file-like object.
        """
        return megfile.smart_open(path, mode)

    @staticmethod
    @deprecated_func("use megfile.smart_copy instead")
    def copy(src_path: str, dst_path: str) -> bool:
        """
        Copies a source path to a destination path.

        Args:
            src_path (str): A URI supported by this PathHandler
            dst_path (str): A URI supported by this PathHandler
            overwrite (bool): Bool flag for forcing overwrite of existing file

        Returns:
            status (bool): True on success
        """
        return megfile.smart_copy(src_path, dst_path)

    @staticmethod
    @deprecated_func("use megfile.smart_realpath instead")
    def get_local_path(path: str) -> str:
        """
        Get a filepath which is compatible with native Python I/O such as `open`
        and `os.path`.

        If URI points to a remote resource, this function may download and cache
        the resource to local disk.

        Args:
            path (str): A URI supported by this PathHandler

        Returns:
            local_path (str): a file path which exists on the local file system
        """
        return megfile.smart_realpath(path)

    @staticmethod
    @deprecated_func("use megfile.smart_exists instead")
    def exists(path: str) -> bool:
        """
        Checks if there is a resource at the given URI.

        Args:
            path (str): A URI supported by this PathHandler

        Returns:
            bool: true if the path exists
        """
        return megfile.smart_exists(path)

    @staticmethod
    @deprecated_func("use megfile.smart_isfile instead")
    def isfile(path: str) -> bool:
        """
        Checks if there the resource at the given URI is a file.

        Args:
            path (str): A URI supported by this PathHandler

        Returns:
            bool: true if the path is a file
        """
        return megfile.smart_isfile(path)

    @staticmethod
    @deprecated_func("use megfile.smart_isdir instead")
    def isdir(path: str) -> bool:
        """
        Checks if the resource at the given URI is a directory.

        Args:
            path (str): A URI supported by this PathHandler

        Returns:
            bool: true if the path is a directory
        """
        return megfile.smart_isdir(path)

    @staticmethod
    @deprecated_func("use megfile.smart_listdir instead")
    def ls(path: str):
        """
        List the contents of the directory at the provided URI.

        Args:
            path (str): A URI supported by this PathHandler

        Returns:
            List[str]: list of contents in given path
        """
        return megfile.smart_listdir(path)

    @staticmethod
    @deprecated_func("use megfile.smart_makedirs instead")
    def mkdirs(path: str, exist_ok: bool = True) -> None:
        """
        Recursive directory creation function. Like mkdir(), but makes all
        intermediate-level directories needed to contain the leaf directory.
        Similar to the native `os.makedirs`.

        Args:
            path (str): A URI supported by this PathHandler
            exist_ok (str): An exception will be raised if dir exist and exist_ok is Flase.
        """
        return megfile.smart_makedirs(path, exist_ok=exist_ok)

    @staticmethod
    @deprecated_func("use megfile.smart_remove instead")
    def rm(path: str) -> None:
        """
        Remove the file (not directory) at the provided URI.

        Args:
            path (str): A URI supported by this PathHandler
        """
        return megfile.smart_remove(path)

    @staticmethod
    @deprecated_func("use megfile.smart_stat instead")
    def stat(path: str):
        """
        get status of the file at the provided URI.

        Args:
            path (str): A URI supported by this PathHandler
        """
        return megfile.smart_stat(path)

    @staticmethod
    @deprecated_func("use megfile.s3_upload instead")
    def upload(local: str, remote: str):
        """
        Upload the local file (not directory) to the specified remote URI.

        Args:
            local (str): path of the local file to be uploaded.
            remote (str): the remote s3uri.
        """
        try:
            megfile.s3_upload(local, remote)
        except Exception:
            return False
        return True

    @staticmethod
    @deprecated_func("use megfile.smart_path_join instead")
    def join(*paths):
        return megfile.smart_path_join(*paths)
