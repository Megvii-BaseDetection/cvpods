
## Getting Started with cvpods

This document provides a brief intro of the usage of builtin command-line tools in cvpods.

For a tutorial that involves actual coding with the API,
see our [playground](https://git-core.megvii-inc.com/zhubenjin/cvpods_playground/tree/master/examples)
which covers how to run inference with an
existing model, and how to train a builtin model on a custom dataset.

For more advanced tutorials, refer to our [documentation](https://luoshu.iap.wh-a.brainpp.cn/docs/cvpods/en/latest/).


### Inference Demo with Pre-trained Models

Our trained model in example is saved in s3://wangfengdata/cvpods_modelzoo,
you may pick a model you want with the following command, for example, maskrcnn baseline
```shell
oss cp \
  s3://wangfengdata/cvpods_modelzoo/detmodel/maskrcnn.res50.fpn.800size.1x.pth .
```

1. Pick a model and get its code link from
	[model zoo](https://git-core.megvii-inc.com/zhubenjin/cvpods/blob/megvii/MODEL_ZOO.md),
2. We provide `demo.py` that is able to run builtin standard models. Run it with:
```
python demo/demo.py \
  --input input1.jpg input2.jpg \
  --opts MODEL.WEIGHTS your_model.pkl
```
The configs are made for training, therefore we need to specify `MODEL.WEIGHTS` to a model from model zoo for evaluation.
This command will run the inference and show visualizations in an OpenCV window.

For details of the command line arguments, see `demo.py -h` or look at its source code
to understand its behavior. Some common arguments are:
* To run __on your webcam__, replace `--input files` with `--webcam`.
* To run __on a video__, replace `--input files` with `--video-input video.mp4`.
* To run __on cpu__, add `MODEL.DEVICE cpu` after `--opts`.
* To save outputs to a directory (for images) or a file (for webcam or video), use `--output`.


### Training & Evaluation in Command Line

We provide a script in "tools/{,plain_}train_net.py", that is made to train
all the configs provided in cvpods.
You may want to use it as a reference to write your own training script for a new research.

To train a model with "train_net.py", first
setup the corresponding datasets following
[datasets/README.md](https://git-core.megvii-inc.com/zhubenjin/cvpods/tree/megvii/datasets),
enter the directory of your code, then run:
```
python path_to_your_cvpods/tools/train_net.py --num-gpus 8
```
To train a model with "cvpods_train", enter the directory and run:
```
cvpods_train --num-gpus 8
```

The configs are made for 8-GPU training. To train on 1 GPU, change the batch size with:
```
cvpods_train --nums-gpu 1 SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025
```
After training, a README.md file is generated automaticly, it shows you the model performance.  
For most models, CPU training is not supported.

(Note that we applied the [linear learning rate scaling rule](https://arxiv.org/abs/1706.02677)
when changing the batch size.)

To evaluate a model's performance, use
```
cvpods_test --nums-gpu 8
```
this will help you to test the performance of latest model. You may set start-iter and end-iter if you want:
```shell
cvpods_test --nums-gpu 8 --start-iter 0 --end-iter 10000
```    
For more options, see `python tools/train_net.py -h`.
