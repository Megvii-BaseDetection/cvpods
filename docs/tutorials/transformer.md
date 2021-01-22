# Extend Data Augmentation

Transformer in cvpods could be divided into two parts:
1. transformers used in config pipeline, named `TransformGen`, such as `CenterAffine`
2. transformers used for given structure, named `Transform`, such as `AffineTransform`

first, you should wirte a `Transform`, then a `TransformGen`. In order to use your transformer,
add it in your train/test pipeline.

### Extend Transform 
inhert `Transform` and override `apply_image`, `apply_coords`, `apply_segmentation` method.  
for example,
```python
class HFlipTransform(Transform):
    """
    Perform horizontal flip.
    """

    def __init__(self, width: int):
        super().__init__()
        self._set_attributes(locals())

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
        tensor = torch.from_numpy(np.ascontiguousarray(img))
        if len(tensor.shape) == 2:
            # For dimension of HxW.
            tensor = tensor.flip((-1))
        elif len(tensor.shape) > 2:
            # For dimension of HxWxC, NxHxWxC.
            tensor = tensor.flip((-2))
        return tensor.numpy()

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Flip the coordinates.

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
        coords[:, 0] = self.width - coords[:, 0]
        return coords
```

### Extend TransformGen  
`TransformGen` creates a `Transform` based on the given image, sometimes with randomness.
To extend TransformGen, you should inherit `TransformGen` and override `get_transform` method. For example,  
```Python
class RandomFlip(TransformGen):

    def __init__(self, prob=0.5):
        """
        Args:
            prob (float): probability of flip.
        """
        horiz, vert = True, False
        # TODO implement vertical flip when we need it
        super().__init__()

        if horiz and vert:
            raise ValueError(
                "Cannot do both horiz and vert. Please use two Flip instead."
            )
        if not horiz and not vert:
            raise ValueError("At least one of horiz or vert has to be True!")
        self._init(locals())

    def get_transform(self, img):
        _, w = img.shape[:2]
        do = self._rand_range() < self.prob
        if do:
            return HFlipTransform(w)
        else:
            return NoOpTransform()
```

### Use transform in config
During train/test, use config.INPUT.AUG to define train/test pipeline, for example,
```python
INPUT=dict(
        AUG=dict(
            TRAIN_PIPELINES=[
                ("ResizeShortestEdge", dict(short_edge_length=(800,), max_size=1333, sample_style="choice")),
                ("RandomFlip", dict()),
            ],
            TEST_PIPELINES=[
                ("ResizeShortestEdge", dict(short_edge_length=800, max_size=1333, sample_style="choice")),
            ],
        )
    )

``` 
which means use Resize and RandomFlip augmentation in training, just resize in testing.