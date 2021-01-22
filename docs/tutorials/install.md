
### Requirements
- Linux or macOS with Python ≥ 3.6
- PyTorch ≥ 1.3
- [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
	You can install them together at [pytorch.org](https://pytorch.org) to make sure of this.
- OpenCV, optional, needed by demo and visualization
- pycocotools: `pip install cython; pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'`
- gcc & g++ ≥ 4.9


### Build cvpods from source

After having the above dependencies and gcc & g++ ≥ 4.9, run:
```
git clone --recursive https://git-core.megvii-inc.com/zhubenjin/cvpods
cd cvpods
rlaunch --cpu 8 --gpu 1 --memory 10240 -- pip3 install -e .
# (add --user if you don't have permission)

# or if you are on macOS
# MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py build develop

# or, as an alternative to `setup.py`, do
# pip install .
```

To __rebuild__ cvpods that's from local clone, `rm -rf build/ **/*.so` then `pip install -e .`.
You often need to rebuild cvpods after reinstalling PyTorch.

### Common Installation Issues

Click each issue for its solutions:

<details>
<summary>
Undefined torch/aten/caffe2 symbols, or segmentation fault immediately when running the library.
</summary>

This can happen if cvpods or torchvision is not
compiled with the version of PyTorch you're running.

If you use a pre-built torchvision, uninstall torchvision & pytorch, and reinstall them
following [pytorch.org](http://pytorch.org).
If you manually build cvpods or torchvision, remove the files you built (`build/`, `**/*.so`)
and rebuild them.

If you cannot resolve the problem, please include the output of `gdb -ex "r" -ex "bt" -ex "quit" --args python -m cvpods.utils.collect_env`
in your issue.
</details>

<details>
<summary>
Undefined C++ symbols in `cvpods/_C*.so`.
</summary>
Usually it's because the library is compiled with a newer C++ compiler but run with an old C++ run time.
This can happen with old anaconda.

Try `conda update libgcc`. Then rebuild cvpods.
</details>

<details>
<summary>
"Not compiled with GPU support" or "cvpods CUDA Compiler: not available".
</summary>
CUDA is not found when building cvpods.
You should make sure
```
python -c 'import torch; from torch.utils.cpp_extension import CUDA_HOME; print(torch.cuda.is_available(), CUDA_HOME)'
```
print valid outputs at the time you build cvpods.
</details>

<details>
<summary>
"invalid device function" or "no kernel image is available for execution".
</summary>

Two possibilities:

* You build cvpods with one version of CUDA but run it with a different version.

To check whether it is the case,
  use `python -m cvpods.utils.collect_env` to find out inconsistent CUDA versions.
	In the output of this command, you should expect "cvpods CUDA Compiler", "CUDA_HOME", "PyTorch built with - CUDA"
	to contain cuda libraries of the same version.

	When they are inconsistent,
	you need to either install a different build of PyTorch (or build by yourself)
	to match your local CUDA installation, or install a different version of CUDA to match PyTorch.

* cvpods or PyTorch/torchvision is not built for the correct GPU architecture (compute compatibility).

	The GPU architecture for PyTorch/cvpods/torchvision is available in the "architecture flags" in
	`python -m cvpods.utils.collect_env`.

	The GPU architecture flags of cvpods/torchvision by default matches the GPU model detected
	during building. This means the compiled code may not work on a different GPU model.
	To overwrite the GPU architecture for cvpods/torchvision, use `TORCH_CUDA_ARCH_LIST` environment variable during building.

	For example, `export TORCH_CUDA_ARCH_LIST=6.0,7.0` makes it work for both P100s and V100s.
    Visit [developer.nvidia.com/cuda-gpus](https://developer.nvidia.com/cuda-gpus) to find out
	the correct compute compatibility number for your device.

</details>

<details>
<summary>
Undefined CUDA symbols.
</summary>

The version of NVCC you use to build cvpods or torchvision does
not match the version of CUDA you are running with.
This often happens when using anaconda's CUDA runtime.

Use `python -m cvpods.utils.collect_env` to find out inconsistent CUDA versions.
In the output of this command, you should expect "cvpods CUDA Compiler", "CUDA_HOME", "PyTorch built with - CUDA"
to contain cuda libraries of the same version.

When they are inconsistent,
you need to either install a different build of PyTorch (or build by yourself)
to match your local CUDA installation, or install a different version of CUDA to match PyTorch.
</details>


<details>
<summary>
"ImportError: cannot import name '_C'".
</summary>
Please build and install cvpods following the instructions above.
</details>
