import inspect
import pprint
import random
import sys
from abc import ABCMeta, abstractmethod
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps
import pycocotools.mask as mask_util

import torch

import cvpods
from cvpods.structures import Boxes, BoxMode, pairwise_iou

from ..registry import TRANSFORMS

__all__ = [
    "Pad",
    "RandomScale",
    "Expand",
    "MinIoURandomCrop",
    "RandomSwapChannels",
    "CenterAffine",
    "RandomBrightness",
    "RandomContrast",
    "RandomCrop",
    "RandomCropWithInstance",
    "RandomCropWithMaxAreaLimit",
    "RandomCropPad",
    "RandomExtent",
    "RandomFlip",
    "RandomSaturation",
    "RandomDistortion",
    "Resize",
    "ResizeShortestEdge",
    "ResizeLongestEdge",
    "ShuffleList",
    "RandomList",
    "RepeatList",
    "TorchTransformGen",
    # transforms used in ssl
    "RandomGaussianBlur",
    "RandomSolarization",
    "RandomLightning",
]


class Transform(metaclass=ABCMeta):
    """
    Base class for implementations of __deterministic__ transformations for
    image and other data structures. "Deterministic" requires that the output of
    all methods of this class are deterministic w.r.t their input arguments. In
    training, there should be a higher-level policy that generates (likely with
    random variations) these transform ops. Each transform op may handle several
    data types, e.g.: image, point cloud, coordinates, segmentation, bounding boxes.
    Some of them have a default implementation, but can be overwritten if the
    default isn't appropriate. The implementation of each method may choose to
    modify its input data in-place for efficient transformation.
    """

    def _set_attributes(self, params: list = None):
        """
        Set attributes from the input list of parameters.

        Args:
            params (list): list of parameters.
        """

        if params is not None:
            for k, v in params.items():
                if k != "self" and not k.startswith("_"):
                    setattr(self, k, v)

    def __call__(self, data: list, annotations: list = None, **kwargs):
        """
        Apply transform to the data and corresponding annotations (if exist).
        """
        raise NotImplementedError

    @classmethod
    def register_type(cls, data_type: str, func: Callable):
        """
        Register the given function as a handler that this transform will use
        for a specific data type.

        Args:
            data_type (str): the name of the data type (e.g., box)
            func (callable): takes a transform and a data, returns the
                transformed data.

        Examples:

        .. code-block:: python

            def func(flip_transform, voxel_data):
                return transformed_voxel_data
            HFlipTransform.register_type("voxel", func)

            # ...
            transform = HFlipTransform(...)
            transform.apply_voxel(voxel_data)  # func will be called
        """
        assert callable(
            func
        ), "You can only register a callable to a Transform. Got {} instead.".format(
            func)
        argspec = inspect.getfullargspec(func)
        assert len(argspec.args) == 2, (
            "You can only register a function that takes two positional "
            "arguments to a Transform! Got a function with spec {}".format(
                str(argspec)))
        setattr(cls, "apply_" + data_type, func)

    def _rand_range(self, low=1.0, high=None, size=None):
        """
        Uniform float random number between low and high.
        """
        if high is None:
            low, high = 0, low
        if size is None:
            size = []
        return np.random.uniform(low, high, size)

    def __repr__(self):
        """
        Produce something like:
        "MyTransformGen(field1={self.field1}, field2={self.field2})"
        """
        try:
            sig = inspect.signature(self.__init__)
            classname = type(self).__name__
            argstr = []
            for name, param in sig.parameters.items():
                assert (
                    param.kind != param.VAR_POSITIONAL
                    and param.kind != param.VAR_KEYWORD
                ), "The default __repr__ doesn't support *args or **kwargs"
                assert hasattr(self, name), (
                    "Attribute {} not found! "
                    "Default __repr__ only works if attributes match the constructor.".format(
                        name
                    )
                )
                attr = getattr(self, name)
                default = param.default
                if default is attr:
                    continue
                argstr.append("{}={}".format(name, pprint.pformat(attr)))
            return "{}({})".format(classname, ", ".join(argstr))
        except AssertionError:
            return super().__repr__()


class ComposeTransform(Transform):
    """
    Composes several transforms together.
    """

    def __init__(self, transforms: list):
        """
        Args:
            transforms (list[Transform]): list of transforms to compose.
        """
        super().__init__()
        self._set_attributes(locals())

    def __eq__(self, other):
        if not isinstance(other, ComposeTransform):
            return False
        return self.transforms == other.transforms

    def __call__(self, img, annotations=None, **kwargs):
        for tfm in self.transforms:
            img, annotations = tfm(img, annotations, **kwargs)
        return img, annotations

    def __repr__(self):
        return "".join([tfm for tfm in self.transforms])


@TRANSFORMS.register()
class RandomList(ComposeTransform):
    """
    Random select subset of provided augmentations.
    """
    def __init__(self, transforms, num_layers=2, choice_weights=None):
        """
        Args:
            transforms (List[TorchTransformGen]): list of transforms need to be performed.
            num_layers (int): parameters of np.random.choice.
            choice_weights (optional, float): parameters of np.random.choice.
        """
        self.all_transforms = transforms
        self.num_layers = num_layers
        self.choice_weights = choice_weights

    def __call__(self, img, annotations=None, **kwargs):
        self.transforms = np.random.choice(
            self.all_transforms,
            self.num_layers,
            replace=self.choice_weights is None,
            p=self.choice_weights)

        return super().__call__(img, annotations)


@TRANSFORMS.register()
class ShuffleList(ComposeTransform):
    """
    Randomly shuffle the `transforms` order.
    """

    def __call__(self, img, annotations=None, **kwargs):
        np.random.shuffle(self.transforms)
        return super().__call__(img, annotations)


@TRANSFORMS.register()
class RepeatList(ComposeTransform):
    """
    Forward several times of provided transforms for a given image.
    """
    def __init__(self, transforms, repeat_times=2):
        """
        Args:
            transforms (list[TransformGen]): List of transform to be repeated.
            repeat_times (int): number of duplicates desired.
        """
        super().__init__(transforms)
        self.times = repeat_times

    def __call__(self, img, annotations=None, **kwargs):
        repeat_imgs = []
        repeat_annotations = []
        for t in range(self.times):
            tmp_img, tmp_anno = super().__call__(img, annotations, **kwargs)
            repeat_imgs.append(tmp_img)
            repeat_annotations.append(tmp_anno)
        repeat_imgs = np.stack(repeat_imgs, axis=0)

        return repeat_imgs, repeat_annotations


class DefaultTransorm(Transform):
    """
    Default transform for 2D detection, segmentation, keypoints, etc.
    """

    @abstractmethod
    def apply_image(self, img: np.ndarray):
        """
        Apply the transform on an image.

        Args:
            img (ndarray): of shape NxHxWxC, or HxWxC or HxW. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].

        Returns:
            ndarray: image after apply the transformation.
        """
        pass

    @abstractmethod
    def apply_coords(self, coords: np.ndarray):
        """
        Apply the transform on coordinates.

        Args:
            coords (ndarray): floating point array of shape Nx2. Each row is (x, y).

        Returns:
            ndarray: coordinates after apply the transformation.

        Note:
            The coordinates are not pixel indices. Coordinates on an image of
            shape (H, W) are in range [0, W] or [0, H].
        """

        pass

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Apply the transform on a full-image segmentation.
        By default will just perform "apply_image".

        Args:
            segmentation (ndarray): of shape HxW. The array should have integer
            or bool dtype.

        Returns:
            ndarray: segmentation after apply the transformation.
        """
        return self.apply_image(segmentation)

    def apply_box(self, box: np.ndarray) -> np.ndarray:
        """
        Apply the transform on an axis-aligned box.
        By default will transform the corner points and use their
        minimum/maximum to create a new axis-aligned box.
        Note that this default may change the size of your box, e.g. in
        rotations.

        Args:
            box (ndarray): Nx4 floating point array of XYXY format in absolute
                coordinates.
        Returns:
            ndarray: box after apply the transformation.

        Note:
            The coordinates are not pixel indices. Coordinates on an image of
            shape (H, W) are in range [0, W] or [0, H].
        """
        # Indexes of converting (x0, y0, x1, y1) box into 4 coordinates of
        # ([x0, y0], [x1, y0], [x0, y1], [x1, y1]).
        idxs = np.array([(0, 1), (2, 1), (0, 3), (2, 3)]).flatten()
        coords = np.asarray(box).reshape(-1, 4)[:, idxs].reshape(-1, 2)
        coords = self.apply_coords(coords).reshape((-1, 4, 2))
        minxy = coords.min(axis=1)
        maxxy = coords.max(axis=1)
        trans_boxes = np.concatenate((minxy, maxxy), axis=1)
        return trans_boxes

    def apply_polygons(self, polygons: list) -> list:
        """
        Apply the transform on a list of polygons, each represented by a Nx2
        array.
        By default will just transform all the points.

        Args:
            polygon (list[ndarray]): each is a Nx2 floating point array of
                (x, y) format in absolute coordinates.
        Returns:
            list[ndarray]: polygon after apply the transformation.

        Note:
            The coordinates are not pixel indices. Coordinates on an image of
            shape (H, W) are in range [0, W] or [0, H].
        """
        return [self.apply_coords(p) for p in polygons]

    def __call__(self, image, annotations=None, **kwargs):
        """
        Apply transfrom to images and annotations (if exist)
        """
        image_size = image.shape[:2]    # h, w
        image = self.apply_image(image)

        if annotations is not None:
            for annotation in annotations:
                if "bbox" in annotation:
                    bbox = BoxMode.convert(
                        annotation["bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS)
                    # Note that bbox is 1d (per-instance bounding box)
                    annotation["bbox"] = self.apply_box([bbox])[0]
                    annotation["bbox_mode"] = BoxMode.XYXY_ABS

                if "segmentation" in annotation:
                    # each instance contains 1 or more polygons
                    segm = annotation["segmentation"]
                    if isinstance(segm, list):
                        # polygons
                        polygons = [np.asarray(p).reshape(-1, 2) for p in segm]
                        annotation["segmentation"] = [
                            p.reshape(-1) for p in self.apply_polygons(polygons)
                        ]
                    elif isinstance(segm, dict):
                        # RLE
                        mask = mask_util.decode(segm)
                        mask = self.apply_segmentation(mask)
                        assert tuple(mask.shape[:2]) == image_size
                        annotation["segmentation"] = mask
                    else:
                        raise ValueError(
                            "Cannot transform segmentation of type '{}'!"
                            "Supported types are: polygons as list[list[float] or ndarray],"
                            " COCO-style RLE as a dict.".format(type(segm)))

                if "keypoints" in annotation:
                    """
                    Transform keypoint annotation of an image.

                    Args:
                        keypoints (list[float]): Nx3 float in cvpods Dataset format.
                        transforms (TransformList):
                        image_size (tuple): the height, width of the transformed image
                        keypoint_hflip_indices (ndarray[int]): see `create_keypoint_hflip_indices`.
                    """
                    # (N*3,) -> (N, 3)
                    keypoints = annotation["keypoints"]
                    keypoints = np.asarray(keypoints, dtype="float64").reshape(-1, 3)
                    keypoints[:, :2] = self.apply_coords(keypoints[:, :2])

                    # This assumes that HorizFlipTransform is the only one that does flip
                    do_hflip = isinstance(self, cvpods.data.transforms.transform.HFlipTransform)

                    # Alternative way: check if probe points was horizontally flipped.
                    # probe = np.asarray([[0.0, 0.0], [image_width, 0.0]])
                    # probe_aug = transforms.apply_coords(probe.copy())
                    # do_hflip = np.sign(probe[1][0] - probe[0][0]) != np.sign(probe_aug[1][0] - probe_aug[0][0])  # noqa

                    # If flipped, swap each keypoint with its opposite-handed equivalent
                    if do_hflip:
                        if "keypoint_hflip_indices" in kwargs:
                            keypoints = keypoints[kwargs["keypoint_hflip_indices"], :]

                    # Maintain COCO convention that if visibility == 0, then x, y = 0
                    # TODO may need to reset visibility for cropped keypoints,
                    # but it does not matter for our existing algorithms
                    keypoints[keypoints[:, 2] == 0] = 0

                    annotation["keypoints"] = keypoints

                # For sem seg task
                if "sem_seg" in annotation:
                    sem_seg = annotation["sem_seg"]
                    if isinstance(sem_seg, np.ndarray):
                        sem_seg = self.apply_segmentation(sem_seg)
                        assert tuple(sem_seg.shape[:2]) == tuple(image.shape[:2]), (
                            f"Image shape is {image.shape[:2]}, "
                            f"but sem_seg shape is {sem_seg.shape[:2]}."
                        )
                        annotation["sem_seg"] = sem_seg
                    else:
                        raise ValueError(
                            "Cannot transform segmentation of type '{}'!"
                            "Supported type is ndarray.".format(type(sem_seg)))
        return image, annotations


# Simplify this to inherent SimpleTransform
@TRANSFORMS.register()
class RandomDistortion(Transform):
    """
    Distort image w.r.t hue, saturation and exposure.
    """

    def __init__(self, hue, saturation, exposure, image_format="BGR", prob=0.5):
        assert image_format in ["RGB", "BGR"]
        super().__init__()
        self._set_attributes(locals())

        self.cvt_code = {
            "RGB": (cv2.COLOR_RGB2HSV, cv2.COLOR_HSV2RGB),
            "BGR": (cv2.COLOR_BGR2HSV, cv2.COLOR_HSV2BGR),
        }[image_format]
        if saturation > 1.0:
            saturation /= 255.  # in range [0, 1]

    def __call__(self, img, annotations=None, **kwargs):
        """
        Args:
            img (ndarray): of shape HxW, HxWxC, or NxHxWxC. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].

        Returns:
            ndarray: the distorted image(s).
        """
        do = self._rand_range() < self.prob
        if do:
            dhue = np.random.uniform(low=-self.hue, high=self.hue)
            dsat = self._rand_scale(self.saturation)
            dexp = self._rand_scale(self.exposure)

            dtype = img.dtype
            img = cv2.cvtColor(img, self.cvt_code[0])
            img = np.asarray(img, dtype=np.float32) / 255.
            img[:, :, 1] *= dsat
            img[:, :, 2] *= dexp
            H = img[:, :, 0] + dhue

            if dhue > 0:
                H[H > 1.0] -= 1.0
            else:
                H[H < 0.0] += 1.0

            img[:, :, 0] = H
            img = (img * 255).clip(0, 255).astype(np.uint8)
            img = cv2.cvtColor(img, self.cvt_code[1])
            img = np.asarray(img, dtype=dtype)

        return img, annotations

    def _rand_scale(self, upper_bound):
        """
        Calculate random scaling factor.

        Args:
            upper_bound (float): range of the random scale.
        Returns:
            random scaling factor (float) whose range is
            from 1 / s to s .
        """
        scale = np.random.uniform(low=1, high=upper_bound)
        if np.random.rand() > 0.5:
            return scale
        return 1 / scale


@TRANSFORMS.register()
class CenterAffine(DefaultTransorm):
    """
    Augmentation from CenterNet
    """
    def __init__(self, boarder, output_size, pad_value=[0, 0, 0], random_aug=True):

        """
        output_size:(w, h)
        """
        super().__init__()
        self._set_attributes(locals())

    def __call__(self, image, annotations, **kwargs):
        self.img_shape = image.shape[:2]
        self.center, self.scale = self.generate_center_and_scale(self.img_shape)
        self.src, self.dst = self.generate_src_and_dst(self.center, self.scale, self.output_size)
        self.affine = cv2.getAffineTransform(np.float32(self.src), np.float32(self.dst))
        return super().__call__(image, annotations)

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Apply AffineTransform for the image(s).

        Args:
            img (ndarray): of shape HxW, HxWxC, or NxHxWxC. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
        Returns:
            ndarray: the image(s) after applying affine transform.
        """
        return cv2.warpAffine(img,
                              self.affine,
                              self.output_size,
                              flags=cv2.INTER_LINEAR,
                              borderValue=self.pad_value)

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Affine the coordinates.

        Args:
            coords (ndarray): floating point array of shape Nx2. Each row is
                (x, y).
        Returns:
            ndarray: the flipped coordinates.

        Note:
            The inputs are floating point coordinates, not pixel indices.
            Therefore they are flipped by `(W - x, H - y)`, not
            `(W - 1 - x, H 1 - y)`.
        """
        # aug_coord (N, 3) shape, self.affine (2, 3) shape
        w, h = self.output_size
        aug_coords = np.column_stack((coords, np.ones(coords.shape[0])))
        coords = np.dot(aug_coords, self.affine.T)
        coords[..., 0] = np.clip(coords[..., 0], 0, w - 1)
        coords[..., 1] = np.clip(coords[..., 1], 0, h - 1)
        return coords


@TRANSFORMS.register()
class RandomFlip(DefaultTransorm):
    """
    Perform horizontal flip.
    """

    def __init__(self, prob=0.5, *, horizontal=True, vertical=False):
        """
        Args:
            prob (float): probability of flip.
            horizontal (boolean): whether to apply horizontal flipping
            vertical (boolean): whether to apply vertical flipping
        """
        super().__init__()

        if horizontal and vertical:
            raise ValueError(
                "Cannot do both horiz and vert. Please use two Flip instead."
            )
        if not horizontal and not vertical:
            raise ValueError("At least one of horiz or vert has to be True!")
        self._set_attributes(locals())

    def __call__(self, image, annotations, **kwargs):
        h, w = image.shape[:2]
        do = self._rand_range() < self.prob

        if self.horizontal:
            self.width = w
        else:
            self.height = h

        if do:
            return super().__call__(image, annotations, **kwargs)
        else:
            return image, annotations

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Flip the image(s).

        Args:
            img (ndarray): of shape HxW, HxWxC, or NxHxWxC. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].

        Returns:
            ndarray: the flipped image(s).
        """
        if self.horizontal:
            tensor = torch.from_numpy(np.ascontiguousarray(img).copy())
            if len(tensor.shape) == 2:
                # For dimension of HxW.
                tensor = tensor.flip((-1))
            elif len(tensor.shape) > 2:
                # For dimension of HxWxC, NxHxWxC.
                tensor = tensor.flip((-2))
        else:
            tensor = torch.from_numpy(np.ascontiguousarray(img).copy())
            if len(tensor.shape) == 2:
                # For dimension of HxW.
                tensor = tensor.flip((-2))
            elif len(tensor.shape) > 2:
                # For dimension of HxWxC, NxHxWxC.
                tensor = tensor.flip((-3))

        return tensor.numpy()

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Flip the coordinates.

        Args:
            coords (ndarray): floating point array of shape Nx2. Each row is (x, y).

        Returns:
            ndarray: the flipped coordinates.

        Note:
            The inputs are floating point coordinates, not pixel indices.
            Therefore they are flipped by `(W - x, H - y)`, not
            `(W - 1 - x, H 1 - y)`.
        """
        if self.horizontal:
            coords[:, 0] = self.width - coords[:, 0]
        else:
            coords[:, 1] = self.height - coords[:, 1]

        return coords


class NoOpTransform(Transform):
    """
    A transform that does nothing.
    """
    def __init__(self):
        super().__init__()

    def __call__(self, data, annotations):
        return data, annotations


@TRANSFORMS.register()
class RandomGaussianBlur(Transform):
    """
    GaussianBlur using PIL.ImageFilter.GaussianBlur
    """
    def __init__(self, sigma, p=1.0):
        """
        Args:
            sigma (List(float)): sigma of gaussian
            p (float): probability of perform this augmentation
        """
        super().__init__()
        self._set_attributes(locals())

    def __call__(self, image, annotations=None, **kwargs):
        if np.random.random() < self.p:
            sigma = random.uniform(self.sigma[0], self.sigma[1])
            img = Image.fromarray(image).filter(ImageFilter.GaussianBlur(radius=sigma))

        return np.array(img), annotations


@TRANSFORMS.register()
class RandomSolarization(Transform):
    def __init__(self, thresh=128, p=0.5):
        super().__init__()
        self._set_attributes(locals)

    def __call__(self, img, annotations=None, **kwargs) -> np.ndarray:
        if np.random.random() < self.p:
            img = np.array(ImageOps.solarize(Image.fromarray(img), self.thresh))

        return img, annotations


@TRANSFORMS.register()
class Pad(DefaultTransorm):
    """
    Pad image with `pad_value` to the specified `target_h` and `target_w`.

    Adds `top` rows of `pad_value` on top, `left` columns of `pad_value` on the left,
    and then pads the image on the bottom and right with `pad_value` until it has
    dimensions `target_h`, `target_w`.

    This op does nothing if `top` and `left` is zero and the image already has size
    `target_h` by `target_w`.
    """

    def __init__(self,
                 top: int,
                 left: int,
                 target_h: int,
                 target_w: int,
                 pad_value=0,
                 seg_value=255,
                 ):
        """
        Args:
            top (int): number of rows of `pad_value` to add on top.
            left (int): number of columns of `pad_value` to add on the left.
            target_h (int): height of output image.
            target_w (int): width of output image.
            pad_value (int): the value used to pad the image.
            seg_value (int): the value used to pad the semantic seg annotaions.
        """
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray, pad_value=None) -> np.ndarray:
        if pad_value is None:
            pad_value = self.pad_value

        if len(img.shape) == 2:  # semantic segmentation mask
            shape = (self.target_h, self.target_w)
        else:
            shape = (self.target_h, self.target_w, 3)

        pad_img = np.full(shape=shape, fill_value=pad_value).astype(img.dtype)

        rest_h = self.target_h - self.top
        rest_w = self.target_w - self.left

        img_h, img_w = img.shape[:2]
        paste_h, paste_w = min(rest_h, img_h), min(rest_w, img_w)
        pad_img[self.top:self.top + paste_h,
                self.left:self.left + paste_w] = img[:paste_h, :paste_w]
        return pad_img

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        coords[:, 0] = coords[:, 0] + self.left
        coords[:, 1] = coords[:, 1] + self.top
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Apply pad transform on the full-image segmentation.

        Args:
            segmentation (ndarray): of shape HxW. The array should have integer
                or bool dtype.
        Returns:
            ndarray: padded segmentation.
        """
        segmentation = self.apply_image(segmentation, pad_value=self.seg_value)
        return segmentation


@TRANSFORMS.register()
class RandomScale(DefaultTransorm):
    """
    Resize the image to a target size.
    """

    def __init__(self, output_size, ratio_range=(0.1, 2), interp="BILINEAR"):

        """
        Args:
            h, w (int): original image size.
            new_h, new_w (int): new image size.
            interp (str): the interpolation method. Options includes:
              * "NEAREST"
              * "BILINEAR"
              * "BICUBIC"
              * "LANCZOS"
              * "HAMMING"
              * "BOX"
        """
        super().__init__()
        self._set_attributes(locals())
        self.min_ratio, self.max_ratio = ratio_range
        if isinstance(self.output_size, int):
            self.output_size = [self.output_size] * 2

        _str_to_pil_interpolation = {
            "NEAREST": Image.NEAREST,
            "BILINEAR": Image.BILINEAR,
            "BICUBIC": Image.BICUBIC,
            "LANCZOS": Image.LANCZOS,
            "HAMMING": Image.HAMMING,
            "BOX": Image.BOX,
        }
        assert (interp in _str_to_pil_interpolation.keys(
        )), "This interpolation mode ({}) is not currently supported!".format(
            interp)
        self.interp = _str_to_pil_interpolation[interp]

    def __call__(self, img, annotations=None, **kwargs):
        h, w = img.shape[:2]
        output_h, output_w = self.output_size

        # 1. Select a random scale factor.
        random_scale_factor = np.random.uniform(self.min_ratio, self.max_ratio)

        scaled_size_h = int(random_scale_factor * output_h)
        scaled_size_w = int(random_scale_factor * output_w)

        # 2. Recompute the accurate scale_factor using rounded scaled image size.
        image_scale_h = scaled_size_h * 1.0 / h
        image_scale_w = scaled_size_w * 1.0 / w
        image_scale = min(image_scale_h, image_scale_w)

        # 3. Select non-zero random offset (x, y) if scaled image is larger than output_size.
        scaled_h = int(h * 1.0 * image_scale)
        scaled_w = int(w * 1.0 * image_scale)

        self.h, self.w, self.new_h, self.new_w = h, w, scaled_h, scaled_w

        return super().__call__(img, annotations)

    def apply_image(self, img: np.ndarray, interp: str = None) -> np.ndarray:
        """
        Resize the image(s).

        Args:
            img (ndarray): of shape NxHxWxC, or HxWxC or HxW. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].

        Returns:
            ndarray: resized image(s).
        """
        # Method 1: second fastest
        # img = cv2.resize(img, (self.new_w, self.new_h), interpolation=cv2.INTER_LINEAR)

        # Method 2: fastest
        pil_image = Image.fromarray(img)
        interp_method = interp if interp is not None else self.interp
        pil_image = pil_image.resize((self.new_w, self.new_h), interp_method)
        img = np.asarray(pil_image)

        return img

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Compute the coordinates after resize.

        Args:
            coords (ndarray): floating point array of shape Nx2. Each row is
                (x, y).
        Returns:
            ndarray: resized coordinates.
        """
        coords[:, 0] = coords[:, 0] * (self.new_w * 1.0 / self.w)
        coords[:, 1] = coords[:, 1] * (self.new_h * 1.0 / self.h)
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Apply resize on the full-image segmentation.

        Args:
            segmentation (ndarray): of shape HxW. The array should have integer
                or bool dtype.
        Returns:
            ndarray: resized segmentation.
        """
        segmentation = self.apply_image(segmentation, interp=Image.NEAREST)
        return segmentation


class IoUCropTransform(DefaultTransorm):
    """
    Perform crop operations on images.

    This crop operation will checks whether the center of each instance's bbox
    is in the cropped image.
    """

    def __init__(self, x0: int, y0: int, w: int, h: int):
        """
        Args:
            x0, y0, w, h (int): crop the image(s) by img[y0:y0+h, x0:x0+w].
        """
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Crop the image(s).

        Args:
            img (ndarray): of shape NxHxWxC, or HxWxC or HxW. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].

        Returns:
            ndarray: cropped image(s).
        """
        if len(img.shape) <= 3:
            return img[self.y0:self.y0 + self.h, self.x0:self.x0 + self.w]
        else:
            return img[..., self.y0:self.y0 + self.h,
                       self.x0:self.x0 + self.w, :]

    def apply_box(self, box: np.ndarray) -> np.ndarray:
        """
        Apply the transform on an axis-aligned box.
        By default will transform the corner points and use their
        minimum/maximum to create a new axis-aligned box.
        Note that this default may change the size of your box, e.g. in
        rotations.

        Args:
            box (ndarray): Nx4 floating point array of XYXY format in absolute
                coordinates.

        Returns:
            ndarray: box after apply the transformation.

        Note:
            The coordinates are not pixel indices. Coordinates on an image of
            shape (H, W) are in range [0, W] or [0, H].
        """
        # Indexes of converting (x0, y0, x1, y1) box into 4 coordinates of
        # ([x0, y0], [x1, y0], [x0, y1], [x1, y1]).
        box = np.array(box).reshape(-1, 4)
        center = (box[:, :2] + box[:, 2:]) / 2
        mask = ((center[:, 0] > self.x0) * (center[:, 0] < self.x0 + self.w)
                * (center[:, 1] > self.y0) * (center[:, 1] < self.y0 + self.h))
        if not mask.any():
            return np.zeros_like(box)

        tl = np.array([self.x0, self.y0])
        box[:, :2] = np.maximum(box[:, :2], tl)
        box[:, :2] -= tl

        box[:, 2:] = np.minimum(box[:, 2:],
                                np.array([self.x0 + self.w, self.y0 + self.h]))
        box[:, 2:] -= tl

        return box

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Apply crop transform on coordinates.

        Args:
            coords (ndarray): floating point array of shape Nx2. Each row is (x, y).

        Returns:
            ndarray: cropped coordinates.
        """
        coords[:, 0] -= self.x0
        coords[:, 1] -= self.y0
        return coords

    def apply_polygons(self, polygons: list) -> list:
        """
        Apply crop transform on a list of polygons, each represented by a Nx2 array.
        It will crop the polygon with the box, therefore the number of points in the
        polygon might change.

        Args:
            polygon (list[ndarray]): each is a Nx2 floating point array of
                (x, y) format in absolute coordinates.

        Returns:
            ndarray: cropped polygons.
        """
        import shapely.geometry as geometry

        # Create a window that will be used to crop
        crop_box = geometry.box(self.x0, self.y0, self.x0 + self.w,
                                self.y0 + self.h).buffer(0.0)

        cropped_polygons = []

        for polygon in polygons:
            polygon = geometry.Polygon(polygon).buffer(0.0)
            # polygon must be valid to perform intersection.
            assert polygon.is_valid, polygon
            cropped = polygon.intersection(crop_box)
            if cropped.is_empty:
                continue
            if not isinstance(cropped,
                              geometry.collection.BaseMultipartGeometry):
                cropped = [cropped]
            # one polygon may be cropped to multiple ones
            for poly in cropped:
                # It could produce lower dimensional objects like lines or
                # points, which we want to ignore
                if not isinstance(poly, geometry.Polygon) or not poly.is_valid:
                    continue
                coords = np.asarray(poly.exterior.coords)
                # NOTE This process will produce an extra identical vertex at
                # the end. So we remove it. This is tested by
                # `tests/test_data_transform.py`
                cropped_polygons.append(coords[:-1])
        return [self.apply_coords(p) for p in cropped_polygons]


@TRANSFORMS.register()
class MinIoURandomCrop(IoUCropTransform):
    """
    Random crop the image & bboxes, the cropped patches have minimum IoU
    requirement with original image & bboxes, the IoU threshold is randomly
    selected from min_ious.
    """

    def __init__(self, min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.3):
        """
        Args:
            min_ious (tuple): minimum IoU threshold for all intersections with bounding boxes
            min_crop_size (float): minimum crop's size
                (i.e. h,w := a*h, a*w, where a >= min_crop_size).
        """
        self._set_attributes(locals())

    def __call__(self, img, annotations=None, **kwargs):
        """
        Args:
            img (ndarray): of shape HxWxC(RGB). The array can be of type uint8
                in range [0, 255], or floating point in range [0, 255].
            annotations (list[dict[str->str]]):
                Each item in the list is a bbox label of an object. The object is
                    represented by a dict,
                which contains:
                 - bbox (list): bbox coordinates, top left and bottom right.
                 - bbox_mode (str): bbox label mode, for example: `XYXY_ABS`,
                    `XYWH_ABS` and so on...
        """
        sample_mode = (1, *self.min_ious, 0)
        h, w = img.shape[:2]

        boxes = list()
        for obj in annotations:
            boxes.append(BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS))
        boxes = torch.tensor(boxes)

        while True:
            mode = np.random.choice(sample_mode)
            if mode == 1:
                return NoOpTransform()

            min_iou = mode
            for i in range(50):
                new_w = np.random.uniform(self.min_crop_size * w, w)
                new_h = np.random.uniform(self.min_crop_size * h, h)

                # h / w in [0.5, 2]
                if new_h / new_w < 0.5 or new_h / new_w > 2:
                    continue

                left = np.random.uniform(w - new_w)
                top = np.random.uniform(h - new_h)

                patch = np.array(
                    (int(left), int(top), int(left + new_w), int(top + new_h)))

                overlaps = pairwise_iou(
                    Boxes(patch.reshape(-1, 4)),
                    Boxes(boxes.reshape(-1, 4))
                )

                if overlaps.min() < min_iou:
                    continue

                # center of boxes should inside the crop img
                center = (boxes[:, :2] + boxes[:, 2:]) / 2
                mask = ((center[:, 0] > patch[0]) * (center[:, 1] > patch[1])
                        * (center[:, 0] < patch[2]) * (center[:, 1] < patch[3]))
                if not mask.any():
                    continue

                self.x0, self.y0, self.w, self.h = int(left), int(top), int(new_w), int(new_h)

                return super().__call__(img, annotations)


class CropTransform(DefaultTransorm):
    """
    Perform crop operations on images.
    """

    def __init__(self, x0: int, y0: int, w: int, h: int):
        """
        Args:
            x0, y0, w, h (int): crop the image(s) by img[y0:y0+h, x0:x0+w].
        """
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Crop the image(s).

        Args:
            img (ndarray): of shape NxHxWxC, or HxWxC or HxW. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].

        Returns:
            ndarray: cropped image(s).
        """
        if len(img.shape) <= 3:
            return img[self.y0:self.y0 + self.h, self.x0:self.x0 + self.w]
        else:
            return img[..., self.y0:self.y0 + self.h,
                       self.x0:self.x0 + self.w, :]

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Apply crop transform on coordinates.

        Args:
            coords (ndarray): floating point array of shape Nx2. Each row is (x, y).

        Returns:
            ndarray: cropped coordinates.
        """
        coords[:, 0] -= self.x0
        coords[:, 1] -= self.y0
        return coords

    def apply_polygons(self, polygons: list) -> list:
        """
        Apply crop transform on a list of polygons, each represented by a Nx2 array.
        It will crop the polygon with the box, therefore the number of points in the
        polygon might change.

        Args:
            polygon (list[ndarray]): each is a Nx2 floating point array of
                (x, y) format in absolute coordinates.

        Returns:
            ndarray: cropped polygons.
        """
        import shapely.geometry as geometry

        # Create a window that will be used to crop
        crop_box = geometry.box(self.x0, self.y0, self.x0 + self.w,
                                self.y0 + self.h).buffer(0.0)

        cropped_polygons = []

        for polygon in polygons:
            polygon = geometry.Polygon(polygon).buffer(0.0)
            # polygon must be valid to perform intersection.
            assert polygon.is_valid, polygon
            cropped = polygon.intersection(crop_box)
            if cropped.is_empty:
                continue
            if not isinstance(cropped,
                              geometry.collection.BaseMultipartGeometry):
                cropped = [cropped]
            # one polygon may be cropped to multiple ones
            for poly in cropped:
                # It could produce lower dimensional objects like lines or
                # points, which we want to ignore
                if not isinstance(poly, geometry.Polygon) or not poly.is_valid:
                    continue
                coords = np.asarray(poly.exterior.coords)
                # NOTE This process will produce an extra identical vertex at
                # the end. So we remove it. This is tested by
                # `tests/test_data_transform.py`
                cropped_polygons.append(coords[:-1])
        return [self.apply_coords(p) for p in cropped_polygons]


@TRANSFORMS.register()
class RandomCrop(CropTransform):
    """
    Randomly crop a subimage out of an image.
    """

    def __init__(self, crop_type: str, crop_size, strict_mode=True):
        """
        Args:
            crop_type (str): one of "relative_range", "relative", "absolute".
                See `config/defaults.py` for explanation.
            crop_size (tuple[float]): the relative ratio or absolute pixels of
                height and width
            strict_mode (bool): if `True`, the target `crop_size` must be smaller than
                the original image size.
        """
        assert crop_type in ["relative_range", "relative", "absolute"]
        self._set_attributes(locals())

    def __call__(self, img, annotations=None, **kwargs):
        h, w = img.shape[:2]
        croph, cropw = self.get_crop_size((h, w))
        if self.strict_mode:
            assert h >= croph and w >= cropw, "Shape computation in {} has bugs.".format(
                self
            )
        offset_range_h = max(h - croph, 0)
        offset_range_w = max(w - cropw, 0)
        self.y0 = np.random.randint(offset_range_h + 1)
        self.x0 = np.random.randint(offset_range_w + 1)

        self.w = cropw
        self.h = croph

        return super().__call__(img, annotations)

    def get_crop_size(self, image_size):
        """
        Args:
            image_size (tuple): height, width

        Returns:
            crop_size (tuple): height, width in absolute pixels
        """
        h, w = image_size
        if self.crop_type == "relative":
            ch, cw = self.crop_size
            return int(h * ch + 0.5), int(w * cw + 0.5)
        elif self.crop_type == "relative_range":
            crop_size = np.asarray(self.crop_size, dtype=np.float32)
            ch, cw = crop_size + np.random.rand(2) * (1 - crop_size)
            return int(h * ch + 0.5), int(w * cw + 0.5)
        elif self.crop_type == "absolute":
            return self.crop_size
        else:
            NotImplementedError("Unknown crop type {}".format(self.crop_type))


@TRANSFORMS.register()
class RandomCropPad(RandomCrop):
    def __init__(self,
                 crop_type: str,
                 crop_size,
                 img_value=None,
                 seg_value=None):
        super().__init__(crop_type, crop_size, strict_mode=False)
        self._set_attributes(locals())

    def __call__(self, img, annotations=None, **kwargs):
        h, w = img.shape[:2]
        croph, cropw = self.get_crop_size((h, w))
        h0 = np.random.randint(h - croph + 1) if h >= croph else 0
        w0 = np.random.randint(w - cropw + 1) if w >= cropw else 0
        dh = min(h, croph)
        dw = min(w, cropw)
        # print(w0, h0, dw, dh)

        self.x0, self.y0, self.w, self.h, self.new_w, self.new_h = w0, h0, dw, dh, cropw, croph
        self.crop_trans = CropTransform(self.x0, self.y0, self.w, self.h)
        pad_top_offset = self.get_pad_offset(self.h, self.new_h)
        pad_left_offset = self.get_pad_offset(self.w, self.new_w)
        self.pad_trans = Pad(
            pad_top_offset, pad_left_offset, self.new_h, self.new_w, self.img_value, self.seg_value)

        return super().__call__(img, annotations)

    def get_pad_offset(self, ori: int, tar: int):
        pad_length = max(tar - ori, 0)
        pad_offset = pad_length // 2
        return pad_offset

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Crop and Pad the image(s).

        Args:
            img (ndarray): of shape NxHxWxC, or HxWxC or HxW. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].

        Returns:
            ndarray: cropped and padded image(s).
        """
        img = self.crop_trans.apply_image(img)
        img = self.pad_trans.apply_image(img)
        return img

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Apply crop and pad transform on coordinates.

        Args:
            coords (ndarray): floating point array of shape Nx2. Each row is
                (x, y).

        Returns:
            ndarray: cropped and padded coordinates.
        """
        coords = self.crop_trans.apply_coords(coords)
        coords = self.pad_trans.apply_coords(coords)
        return coords

    def apply_polygons(self, polygons: list) -> list:
        """
        Apply crop and pad transform on a list of polygons, each represented by a Nx2 array.

        Args:
            polygon (list[ndarray]): each is a Nx2 floating point array of
                (x, y) format in absolute coordinates.

        Returns:
            ndarray: cropped and padded polygons.
        """
        polygons = self.crop_trans.apply_polygons(polygons)
        polygons = self.pad_trans.apply_polygons(polygons)
        return polygons

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Apply crop and pad transform on the full-image segmentation.

        Args:
            segmentation (ndarray): of shape HxW. The array should have integer
                or bool dtype.

        Returns:
            ndarray: cropped and padded segmentation.
        """
        segmentation = self.crop_trans.apply_segmentation(segmentation)
        segmentation = self.pad_trans.apply_segmentation(segmentation)
        return segmentation


@TRANSFORMS.register()
class RandomCropWithInstance(RandomCrop):
    """
    Make sure the cropping region contains the center of a random instance from annotations.
    """

    def __call__(self, img, annotations=None, **kwargs):
        h, w = img.shape[:2]
        croph, cropw = self.get_crop_size((h, w))
        if self.strict_mode:
            assert h >= croph and w >= cropw, "Shape computation in {} has bugs.".format(
                self
            )
        offset_range_h = max(h - croph, 0)
        offset_range_w = max(w - cropw, 0)
        # Make sure there is always at least one instance in the image
        assert annotations is not None, "Can not get annotations infos."
        instance = np.random.choice(annotations)
        bbox = BoxMode.convert(instance["bbox"], instance["bbox_mode"], BoxMode.XYXY_ABS)
        bbox = torch.tensor(bbox)
        center_xy = (bbox[:2] + bbox[2:]) / 2.0

        offset_range_h_min = max(center_xy[1] - croph, 0)
        offset_range_w_min = max(center_xy[0] - cropw, 0)
        offset_range_h_max = min(offset_range_h, center_xy[1] - 1)
        offset_range_w_max = min(offset_range_w, center_xy[0] - 1)

        self.y0 = np.random.randint(offset_range_h_min, offset_range_h_max + 1)
        self.x0 = np.random.randint(offset_range_w_min, offset_range_w_max + 1)

        self.w = cropw
        self.h = croph

        return super().__call__(img, annotations)


@TRANSFORMS.register()
class RandomCropWithMaxAreaLimit(RandomCrop):
    """
    Find a cropping window such that no single category occupies more than
    `single_category_max_area` in `sem_seg`.

    The function retries random cropping 10 times max.
    """

    def __init__(self, crop_type: str, crop_size, strict_mode=True,
                 single_category_max_area=1.0, ignore_value=255):
        super().__init__(crop_type, crop_size, strict_mode)
        self._set_attributes(locals())

    def __call__(self, img, annotations=None, **kwargs):
        if self.single_category_max_area >= 1.0:
            return super().__call__(img, annotations)
        else:
            h, w = img.shape[:2]
            assert "sem_seg" in annotations[0]
            sem_seg = annotations[0]["sem_seg"]
            croph, cropw = self.get_crop_size((h, w))
            for _ in range(10):
                y0 = np.random.randint(h - croph + 1)
                x0 = np.random.randint(w - cropw + 1)
                sem_seg_temp = sem_seg[y0: y0 + croph, x0: x0 + cropw]
                labels, cnt = np.unique(sem_seg_temp, return_counts=True)
                cnt = cnt[labels != self.ignore_value]
                if len(cnt) > 1 and np.max(cnt) / np.sum(cnt) < self.single_category_max_area:
                    break
            self.x0, self.y0, self.w, self.h = x0, y0, cropw, croph

            return super().__call__(img, annotations)


class BlendTransform(Transform):
    """
    Transforms pixel colors with PIL enhance functions.
    """

    def __init__(self, src_image: np.ndarray, src_weight: float,
                 dst_weight: float):
        """
        Blends the input image (dst_image) with the src_image using formula:
        ``src_weight * src_image + dst_weight * dst_image``

        Args:
            src_image (ndarray): Input image is blended with this image
            src_weight (float): Blend weighting of src_image
            dst_weight (float): Blend weighting of dst_image
        """
        super().__init__()
        self._set_attributes(locals())

    # def __call__(self, img: np.ndarray, interp: str = None) -> np.ndarray:
    def __call__(self, img, annotations=None, **kwargs):
        """
        Apply blend transform on the image(s).

        Args:
            img (ndarray): of shape NxHxWxC, or HxWxC or HxW. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
            interp (str): keep this option for consistency, perform blend would not
                require interpolation.

        Returns:
            ndarray: blended image(s).
        """
        if img.dtype == np.uint8:
            img = img.astype(np.float32)
            img = self.src_weight * self.src_image + self.dst_weight * img
            return np.clip(img, 0, 255).astype(np.uint8), annotations
        else:
            return self.src_weight * self.src_image + self.dst_weight * img, annotations


@TRANSFORMS.register()
class RandomContrast(BlendTransform):
    """
    Randomly transforms image contrast.

    Contrast intensity is uniformly sampled in (intensity_min, intensity_max).
    - intensity < 1 will reduce contrast
    - intensity = 1 will preserve the input image
    - intensity > 1 will increase contrast

    See: https://pillow.readthedocs.io/en/3.0.x/reference/ImageEnhance.html
    """

    def __init__(self, intensity_min, intensity_max, prob=1.0):
        """
        Args:
            intensity_min (float): Minimum augmentation.
            intensity_max (float): Maximum augmentation.
            prob (float): probability of transforms image contrast.
        """
        super().__init__()
        self._set_attributes(locals())

    def __call__(self, img, annotations=None, **kwargs):
        if self._rand_range() < self.prob:
            w = np.random.uniform(self.intensity_min, self.intensity_max)
            self.src_image, self.src_weight, self.dst_weight = img.mean(), 1 - w, w
            return super().__call__(img, annotations)
        else:
            return img, annotations


@TRANSFORMS.register()
class RandomBrightness(BlendTransform):
    """
    Randomly transforms image brightness.

    Brightness intensity is uniformly sampled in (intensity_min, intensity_max).
    - intensity < 1 will reduce brightness
    - intensity = 1 will preserve the input image
    - intensity > 1 will increase brightness

    See: https://pillow.readthedocs.io/en/3.0.x/reference/ImageEnhance.html
    """

    def __init__(self, intensity_min, intensity_max, prob=1.):
        """
        Args:
            intensity_min (float): Minimum augmentation.
            intensity_max (float): Maximum augmentation.
            prob (float): probability of transforms image brightness.
        """
        super().__init__()
        self._set_attributes(locals())

    def __call__(self, img, annotations=None, **kwargs):
        do = self._rand_range() < self.prob
        if do:
            w = np.random.uniform(self.intensity_min, self.intensity_max)
            self.src_image, self.src_weight, self.dst_weight = 0, 1 - w, w
            return super().__call__(img, annotations)
        else:
            return img, annotations


@TRANSFORMS.register()
class RandomSaturation(BlendTransform):
    """
    Randomly transforms image saturation.

    Saturation intensity is uniformly sampled in (intensity_min, intensity_max).
    - intensity < 1 will reduce saturation (make the image more grayscale)
    - intensity = 1 will preserve the input image
    - intensity > 1 will increase saturation

    See: https://pillow.readthedocs.io/en/3.0.x/reference/ImageEnhance.html
    """

    def __init__(self, intensity_min, intensity_max, prob=1.0):
        """
        Args:
            intensity_min (float): Minimum augmentation (1 preserves input).
            intensity_max (float): Maximum augmentation (1 preserves input).
            prob (float): probability of transforms image saturation.
        """
        super().__init__()
        self._set_attributes(locals())

    def __call__(self, img, annotations=None, **kwargs):
        do = self._rand_range() < self.prob
        if do:
            assert img.shape[-1] == 3, "Saturation only works on RGB images"
            w = np.random.uniform(self.intensity_min, self.intensity_max)
            grayscale = img.dot([0.299, 0.587, 0.114])[:, :, np.newaxis]
            self.src_image, self.src_weight, self.dst_weight = grayscale, 1 - w, w
            return super().__call__(img, annotations)
        else:
            return img, annotations


@TRANSFORMS.register()
class RandomLightning(BlendTransform):
    """
    Randomly transforms image color using fixed PCA over ImageNet.

    The degree of color jittering is randomly sampled via a normal distribution,
    with standard deviation given by the scale parameter.
    """

    def __init__(self, scale, prob=0.5):
        """
        Args:
            scale (float): Standard deviation of principal component weighting.
        """
        super().__init__()
        self._set_attributes(locals())
        self.eigen_vecs = np.array(
            [
                [-0.5675, 0.7192, 0.4009],
                [-0.5808, -0.0045, -0.8140],
                [-0.5836, -0.6948, 0.4203],
            ]
        )
        self.eigen_vals = np.array([0.2175, 0.0188, 0.0045])

    def __call__(self, img, annotations=None, **kwargs):
        assert img.shape[-1] == 3, "Saturation only works on RGB images"
        do = self._rand_range() < self.prob
        if do:
            weights = np.random.normal(scale=self.scale, size=3)
            self.src_image, self.src_weight, self.dst_weight = \
                self.eigen_vecs.dot(weights * self.eigen_vals), 1, 1
            return super().__call__(img, annotations)
        else:
            return img, annotations


@TRANSFORMS.register()
class RandomSwapChannels(Transform):
    """
    Randomly swap image channels.
    """

    def __init__(self, prob=0.5):
        super().__init__()
        self._set_attributes()

    def __call__(self, img, annotations=None, **kwargs):
        assert len(img.shape) > 2
        if self._rand_range() < self.prob:
            return img[..., np.random.permutation(3)], annotations
        else:
            return img, annotations


@TRANSFORMS.register()
class Expand(DefaultTransorm):
    """
    Expand the image and boxes according the specified expand ratio.
    """

    def __init__(self, ratio_range=(1, 4), mean=(0, 0, 0), prob=0.5):

        """
        Args:
            left, top (int): crop the image by img[top: top+h, left:left+w].
            ratio (float): image expand ratio.
            mean (tuple): mean value of dataset.
        """
        super().__init__()
        self._set_attributes(locals())
        self.min_ratio, self.max_ratio = ratio_range

    def __call__(self, img, annotations=None, **kwargs):
        if self._rand_range() < self.prob:
            return img, annotations
        else:
            h, w, c = img.shape
            ratio = np.random.uniform(self.min_ratio, self.max_ratio)
            left = int(np.random.uniform(0, w * ratio - w))
            top = int(np.random.uniform(0, h * ratio - h))
            self.left, self.top, self.ratio = left, top, ratio
            return super().__call__(img, annotations)

    def apply_image(self, img):
        """
        Randomly place the original image on a canvas of 'ratio' x original image
        size filled with mean values. The ratio is in the range of ratio_range.
        """
        h, w, c = img.shape
        expand_img = np.full((int(h * self.ratio), int(w * self.ratio), c),
                             self.mean).astype(img.dtype)
        expand_img[self.top:self.top + h, self.left:self.left + w] = img
        return expand_img

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Apply expand transform on coordinates.

        Args:
            coords (ndarray): floating point array of shape Nx2. Each row is
                (x, y).

        Returns:
            ndarray: expand coordinates.
        """
        coords[:, 0] += self.left
        coords[:, 1] += self.top
        return coords


@TRANSFORMS.register()
class RandomExtent(DefaultTransorm):
    """
    Extracts a subregion from the source image and scales it to the output size.

    The fill color is used to map pixels from the source rect that fall outside
    the source image.

    See: https://pillow.readthedocs.io/en/latest/PIL.html#PIL.ImageTransform.ExtentTransform
    """
    def __init__(self, scale_range, shift_range, interp=Image.LINEAR, fill=0, prob=0.5):
        """
        Args:
            scale_range (l, h): Range of input-to-output size scaling factor.
            shift_range (x, y): Range of shifts of the cropped subrect. The rect
                is shifted by [w / 2 * Uniform(-x, x), h / 2 * Uniform(-y, y)],
                where (w, h) is the (width, height) of the input image. Set each
                component to zero to crop at the image's center.
        """
        super().__init__()
        self._set_attributes(locals())

    def __call__(self, img, annotations=None, **kwargs):

        if self._rand_range() < self.prob:
            return img, annotations
        else:
            img_h, img_w = img.shape[:2]

            # Initialize src_rect to fit the input image.
            src_rect = np.array([-0.5 * img_w, -0.5 * img_h, 0.5 * img_w, 0.5 * img_h])

            # Apply a random scaling to the src_rect.
            src_rect *= np.random.uniform(self.scale_range[0], self.scale_range[1])

            # Apply a random shift to the coordinates origin.
            src_rect[0::2] += self.shift_range[0] * img_w * (np.random.rand() - 0.5)
            src_rect[1::2] += self.shift_range[1] * img_h * (np.random.rand() - 0.5)

            # Map src_rect coordinates into image coordinates (center at corner).
            src_rect[0::2] += 0.5 * img_w
            src_rect[1::2] += 0.5 * img_h

            self.src_rect = (src_rect[0], src_rect[1], src_rect[2], src_rect[3])
            self.output_size = (
                int(src_rect[3] - src_rect[1]),
                int(src_rect[2] - src_rect[0]),
            )

            return super().__call__(img, annotations)

    def apply_image(self, img, interp=None):
        h, w = self.output_size
        ret = Image.fromarray(img).transform(
            size=(w, h),
            method=Image.EXTENT,
            data=self.src_rect,
            resample=interp if interp else self.interp,
            fill=self.fill,
        )
        return np.asarray(ret)

    def apply_coords(self, coords):
        # Transform image center from source coordinates into output coordinates
        # and then map the new origin to the corner of the output image.
        h, w = self.output_size
        x0, y0, x1, y1 = self.src_rect
        new_coords = coords.astype(np.float32)
        new_coords[:, 0] -= 0.5 * (x0 + x1)
        new_coords[:, 1] -= 0.5 * (y0 + y1)
        new_coords[:, 0] *= w / (x1 - x0)
        new_coords[:, 1] *= h / (y1 - y0)
        new_coords[:, 0] += 0.5 * w
        new_coords[:, 1] += 0.5 * h
        return new_coords

    def apply_segmentation(self, segmentation):
        segmentation = self.apply_image(segmentation, interp=Image.NEAREST)
        return segmentation


class ResizeTransform(DefaultTransorm):
    """
    Resize the image to a target size.
    """
    def __init__(self, h, w, new_h, new_w, interp):
        """
        Args:
            h, w (int): original image size
            new_h, new_w (int): new image size
            interp: PIL interpolation methods
        """
        # TODO decide on PIL vs opencv
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img, interp=None):
        assert img.shape[:2] == (self.h, self.w)
        pil_image = Image.fromarray(img)
        interp_method = interp if interp is not None else self.interp
        pil_image = pil_image.resize((self.new_w, self.new_h), interp_method)
        ret = np.asarray(pil_image)
        return ret

    def apply_coords(self, coords):
        coords[:, 0] = coords[:, 0] * (self.new_w * 1.0 / self.w)
        coords[:, 1] = coords[:, 1] * (self.new_h * 1.0 / self.h)
        return coords

    def apply_segmentation(self, segmentation):
        segmentation = self.apply_image(segmentation, interp=Image.NEAREST)
        return segmentation


@TRANSFORMS.register()
class Resize(ResizeTransform):
    """
    Resize image to a target size
    """

    def __init__(self, shape, interp=Image.BILINEAR):
        """
        Args:
            shape: (h, w) tuple or a int.
            interp: PIL interpolation method.
        """
        if isinstance(shape, int):
            shape = (shape, shape)
        shape = tuple(shape)
        self._set_attributes(locals())

    def __call__(self, img, annotations=None, **kwargs):
        self.h, self.w, self.new_h, self.new_w = \
            img.shape[0], img.shape[1], self.shape[0], self.shape[1]
        return super().__call__(img, annotations)


@TRANSFORMS.register()
class ResizeLongestEdge(ResizeTransform):
    """
    Scale the longer edge to the given size.
    """

    def __init__(self, long_edge_length, sample_style="range", interp=Image.BILINEAR,
                 jitter=(0.0, 32)):
        """
        Args:
            long_edge_length (list[int]): If ``sample_style=="range"``,
                a [min, max] interval from which to sample the shortest edge length.
                If ``sample_style=="choice"``, a list of shortest edge lengths to sample from.
            sample_style (str): either "range" or "choice".
            interp: PIL interpolation method.
        """
        assert sample_style in ["range", "choice"], sample_style

        self.is_range = sample_style == "range"
        if isinstance(long_edge_length, int):
            long_edge_length = (long_edge_length, long_edge_length)
        self._set_attributes(locals())

    def __call__(self, img, annotations=None, **kwargs):
        h, w = img.shape[:2]
        if self.is_range:
            size = np.random.randint(
                self.long_edge_length[0], self.long_edge_length[1] + 1
            )
        else:
            size = np.random.choice(self.long_edge_length)
        if size == 0:
            return NoOpTransform()

        if self.jitter[0] > 0:
            dw = self.jitter[0] * w
            dh = self.jitter[0] * h
            size = max(h, w) + np.random.uniform(low=-max(dw, dh), high=max(dw, dh))
            size -= size % self.jitter[1]

        scale = size * 1.0 / max(h, w)
        if h < w:
            newh, neww = scale * h, size
        else:
            newh, neww = size, scale * w

        neww = int(neww + 0.5)
        newh = int(newh + 0.5)

        self.h, self.w, self.new_h, self.new_w = h, w, newh, neww
        return super().__call__(img, annotations)


@TRANSFORMS.register()
class ResizeShortestEdge(ResizeTransform):
    """
    Scale the shorter edge to the given size, with a limit of `max_size` on the longer edge.
    If `max_size` is reached, then downscale so that the longer edge does not exceed max_size.
    """

    def __init__(
        self,
        short_edge_length,
        max_size=sys.maxsize,
        sample_style="range",
        interp=Image.BILINEAR,
    ):
        """
        Args:
            short_edge_length (list[int]): If ``sample_style=="range"``,
                a [min, max] interval from which to sample the shortest edge length.
                If ``sample_style=="choice"``, a list of shortest edge lengths to sample from.
            max_size (int): maximum allowed longest edge length.
            sample_style (str): either "range" or "choice".
            interp: PIL interpolation method.
        """
        assert sample_style in ["range", "choice"], sample_style

        self.is_range = sample_style == "range"
        if isinstance(short_edge_length, int):
            short_edge_length = (short_edge_length, short_edge_length)
        self._set_attributes(locals())

    def __call__(self, img, annotations=None, **kwargs):
        h, w = img.shape[:2]

        if self.is_range:
            size = np.random.randint(
                self.short_edge_length[0], self.short_edge_length[1] + 1
            )
        else:
            size = np.random.choice(self.short_edge_length)
        if size == 0:
            return NoOpTransform()

        scale = size * 1.0 / min(h, w)
        if h < w:
            newh, neww = size, scale * w
        else:
            newh, neww = scale * h, size
        if max(newh, neww) > self.max_size:
            scale = self.max_size * 1.0 / max(newh, neww)
            newh = newh * scale
            neww = neww * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)

        self.h, self.w, self.new_h, self.new_w = h, w, newh, neww
        return super().__call__(img, annotations)


def HFlip_rotated_box(transform, rotated_boxes):
    """
    Apply the horizontal flip transform on rotated boxes.

    Args:
        rotated_boxes (ndarray): Nx5 floating point array of
            (x_center, y_center, width, height, angle_degrees) format
            in absolute coordinates.
    """
    # Transform x_center
    rotated_boxes[:, 0] = transform.width - rotated_boxes[:, 0]
    # Transform angle
    rotated_boxes[:, 4] = -rotated_boxes[:, 4]
    return rotated_boxes


def Resize_rotated_box(transform, rotated_boxes):
    """
    Apply the resizing transform on rotated boxes. For details of how these (approximation)
    formulas are derived, please refer to :meth:`RotatedBoxes.scale`.

    Args:
        rotated_boxes (ndarray): Nx5 floating point array of
            (x_center, y_center, width, height, angle_degrees) format
            in absolute coordinates.
    """
    scale_factor_x = transform.new_w * 1.0 / transform.w
    scale_factor_y = transform.new_h * 1.0 / transform.h
    rotated_boxes[:, 0] *= scale_factor_x
    rotated_boxes[:, 1] *= scale_factor_y
    theta = rotated_boxes[:, 4] * np.pi / 180.0
    c = np.cos(theta)
    s = np.sin(theta)
    rotated_boxes[:, 2] *= np.sqrt(
        np.square(scale_factor_x * c) + np.square(scale_factor_y * s))
    rotated_boxes[:, 3] *= np.sqrt(
        np.square(scale_factor_x * s) + np.square(scale_factor_y * c))
    rotated_boxes[:, 4] = np.arctan2(scale_factor_x * s,
                                     scale_factor_y * c) * 180 / np.pi

    return rotated_boxes


# RandomFlip is horizontal by default.
RandomFlip.register_type("rotated_box", HFlip_rotated_box)
NoOpTransform.register_type("rotated_box", lambda t, x: x)
ResizeTransform.register_type("rotated_box", Resize_rotated_box)


@TRANSFORMS.register()
class TorchTransformGen(Transform):
    """
    Wrapper transfrom of transforms in torchvision.
    It convert img (np.ndarray) to PIL image, and convert back to np.ndarray after transform.
    """
    def __init__(self, tfm):
        self.tfm = tfm

    def __call__(self, img: np.ndarray, annotations: None, **kwargs):
        pil_image = Image.fromarray(img)
        return np.array(self.tfm(pil_image)), annotations
